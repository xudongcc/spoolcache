"""Fail-closed checks for the public vLLM interfaces SpoolCache uses."""

from __future__ import annotations

import hashlib
import inspect
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..errors import UnsupportedRuntimeError
from ..identity import sha256_json

# Public KVConnectorBase_V1 callbacks that SpoolCache overrides or relies on.
# The value is (positional prefix, required **kwargs name). Keeping this table
# independent of model identity makes startup compatibility a runtime API
# decision rather than a per-model allowlist.
_CONNECTOR_HOOK_CONTRACTS: Mapping[
    str, tuple[tuple[str, ...], str | None]
] = {
    "register_kv_caches": (("self", "kv_caches"), None),
    "start_load_kv": (("self", "forward_context"), "kwargs"),
    "wait_for_layer_load": (("self", "layer_name"), None),
    "save_kv_layer": (
        ("self", "layer_name", "kv_layer", "attn_metadata"),
        "kwargs",
    ),
    "wait_for_save": (("self",), None),
    "get_num_new_matched_tokens": (
        ("self", "request", "num_computed_tokens"),
        None,
    ),
    "on_new_request": (("self", "request"), None),
    "update_state_after_alloc": (
        ("self", "request", "blocks", "num_external_tokens"),
        None,
    ),
    "build_connector_meta": (("self", "scheduler_output"), None),
    "get_finished": (("self", "finished_req_ids"), None),
    "get_block_ids_with_load_errors": (("self",), None),
    "get_kv_connector_stats": (("self",), None),
    # getattr() binds this classmethod, so ``cls`` is intentionally absent.
    "build_kv_connector_stats": (("data",), None),
    # vLLM constructs this exporter in the API process. getattr() binds the
    # classmethod, so ``cls`` is intentionally absent here too.
    "build_prom_metrics": (
        (
            "vllm_config",
            "metric_types",
            "labelnames",
            "per_engine_labelvalues",
        ),
        None,
    ),
    "update_connector_output": (("self", "connector_output"), None),
    "get_handshake_metadata": (("self",), None),
    "set_xfer_handshake_metadata": (("self", "metadata"), None),
    "shutdown": (("self",), None),
}

_PP_AWARE_HANDSHAKE_CONTRACT = {
    "set_xfer_handshake_metadata_pp_aware": (("self", "metadata"), None),
}

_FINGERPRINT_KIND = "installed-package-content-sha256"
_IGNORED_PACKAGE_PARTS = frozenset({".git", "__pycache__"})
_IGNORED_PACKAGE_SUFFIXES = frozenset({".pyc", ".pyo"})


@dataclass(frozen=True)
class RuntimeReceipt:
    vllm_version: str
    vllm_build_sha256: str
    build_fingerprint_kind: str
    compatibility_mode: str
    connector_constructor: tuple[str, ...]
    supports_hma_parameters: tuple[str, ...]
    capabilities: tuple[str, ...]

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


def _inspect_signature(callable_object: object, *, label: str) -> inspect.Signature:
    try:
        return inspect.signature(callable_object)
    except (TypeError, ValueError) as error:
        raise UnsupportedRuntimeError(f"cannot inspect {label}") from error


def _verify_spec_kind_resolver(
    resolver: Callable[[object], object] | object | None,
) -> Callable[[object], object]:
    """Require the public cache-spec classifier and its one-argument shape."""

    if resolver is None:
        try:
            cache_interface = import_module("vllm.v1.kv_cache_interface")
            resolver = getattr(cache_interface, "get_kv_cache_spec_kind")
        except (ImportError, AttributeError) as error:
            raise UnsupportedRuntimeError(
                "vLLM lacks the public KV cache semantic-kind resolver"
            ) from error
    if not callable(resolver):
        raise UnsupportedRuntimeError(
            "vLLM KV cache semantic-kind resolver is not callable"
        )
    signature = _inspect_signature(
        resolver,
        label="vLLM KV cache semantic-kind resolver",
    )
    try:
        signature.bind(object())
    except TypeError as error:
        raise UnsupportedRuntimeError(
            "vLLM KV cache semantic-kind resolver cannot accept one cache spec"
        ) from error
    return resolver


def _accepts_required_prefix(
    callable_object: object,
    required: tuple[str, ...],
    *,
    label: str,
    variadic_keyword: str | None = None,
) -> tuple[str, ...]:
    """Validate the stable semantic prefix and reject new mandatory inputs."""

    signature = _inspect_signature(callable_object, label=label)
    parameters = tuple(signature.parameters.values())
    names = tuple(parameter.name for parameter in parameters)
    prefix = parameters[: len(required)]
    if names[: len(required)] != required or any(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in prefix
    ):
        raise UnsupportedRuntimeError(f"{label} contract differs")
    extension = parameters[len(required) :]
    if variadic_keyword is not None:
        if (
            not extension
            or extension[-1].name != variadic_keyword
            or extension[-1].kind is not inspect.Parameter.VAR_KEYWORD
        ):
            raise UnsupportedRuntimeError(f"{label} contract differs")
        extension = extension[:-1]
    for parameter in extension:
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.default is inspect.Parameter.empty:
            raise UnsupportedRuntimeError(f"{label} adds a required parameter")
    return names


def _minimum_call_shape(
    signature: inspect.Signature,
) -> tuple[tuple[object, ...], dict[str, object]]:
    args: list[object] = []
    kwargs: dict[str, object] = {}
    for parameter in signature.parameters.values():
        if parameter.default is not inspect.Parameter.empty:
            continue
        value = object()
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            args.append(value)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = value
    return tuple(args), kwargs


def _maximal_positional_call_shape(
    signature: inspect.Signature,
) -> tuple[tuple[object, ...], dict[str, object]]:
    args: list[object] = []
    kwargs: dict[str, object] = {}
    for parameter in signature.parameters.values():
        value = object()
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            args.append(value)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = value
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            args.extend((object(), object()))
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            kwargs["__spoolcache_future_keyword__"] = object()
    return tuple(args), kwargs


def _maximal_keyword_call_shape(
    signature: inspect.Signature,
) -> tuple[tuple[object, ...], dict[str, object]]:
    args: list[object] = []
    kwargs: dict[str, object] = {}
    for parameter in signature.parameters.values():
        value = object()
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        elif parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            kwargs[parameter.name] = value
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            # A call cannot supply preceding positional-or-keyword parameters
            # by name and also skip over them to reach *args. The fully
            # positional probe above covers arbitrary positional extensions.
            continue
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            kwargs["__spoolcache_future_keyword__"] = object()
    return tuple(args), kwargs


def _assert_override_substitutable(
    base_callable: object,
    override_callable: object,
    *,
    label: str,
) -> None:
    """Prove the override accepts every parameter shape exposed by the base.

    Values and types are deliberately irrelevant here; vLLM's public Python
    signature defines how a caller may supply arguments. The minimum, fully
    positional, fully keyword, ``*args`` and arbitrary ``**kwargs`` shapes
    cover that finite calling convention. A new base option is accepted only
    if the installed SpoolCache override can receive it as declared.
    """

    base_signature = _inspect_signature(base_callable, label=f"{label} base")
    override_signature = _inspect_signature(
        override_callable, label=f"{label} override"
    )
    base_parameters = tuple(base_signature.parameters.values())
    override_parameters = tuple(override_signature.parameters.values())
    if any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in base_parameters
    ) and not any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in override_parameters
    ):
        raise UnsupportedRuntimeError(
            f"{label} override cannot accept vLLM call shape (*args)"
        )
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in base_parameters
    ) and not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in override_parameters
    ):
        raise UnsupportedRuntimeError(
            f"{label} override cannot accept vLLM call shape (**kwargs)"
        )

    call_shapes = (
        _minimum_call_shape(base_signature),
        _maximal_positional_call_shape(base_signature),
        _maximal_keyword_call_shape(base_signature),
    )
    for args, kwargs in call_shapes:
        # Assert each generated shape is genuinely legal for the base before
        # using it as evidence against the override.
        try:
            base_signature.bind(*args, **kwargs)
        except TypeError as error:  # pragma: no cover - internal invariant
            raise AssertionError("generated an invalid base call shape") from error
        try:
            override_signature.bind(*args, **kwargs)
        except TypeError as error:
            raise UnsupportedRuntimeError(
                f"{label} override cannot accept vLLM call shape"
            ) from error

    # Binding alone would miss an override that exposes the same optional
    # names in a different positional order: both positional and keyword calls
    # are legal, but positional values reach the wrong semantic parameter.
    positional_parameters = tuple(
        parameter
        for parameter in base_parameters
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    )
    markers = tuple(object() for _ in positional_parameters)
    base_bound = base_signature.bind_partial(*markers)
    override_bound = override_signature.bind_partial(*markers)
    override_by_name = dict(override_signature.parameters)
    for parameter in positional_parameters:
        marker = base_bound.arguments[parameter.name]
        destination: str | None = None
        for name, bound_value in override_bound.arguments.items():
            if bound_value is marker:
                destination = name
                break
            if isinstance(bound_value, tuple) and any(
                item is marker for item in bound_value
            ):
                destination = name
                break
        if destination is None:  # pragma: no cover - bind invariant
            raise AssertionError("override binding lost a positional argument")
        destination_parameter = override_by_name[destination]
        if (
            destination != parameter.name
            and destination_parameter.kind is not inspect.Parameter.VAR_POSITIONAL
        ):
            raise UnsupportedRuntimeError(
                f"{label} override reorders vLLM positional parameters"
            )


def _package_roots(
    vllm_module: object,
    supplied_roots: Sequence[Path] | None,
) -> tuple[Path, ...]:
    raw_roots: Sequence[object]
    if supplied_roots is not None:
        raw_roots = supplied_roots
    else:
        module_path = getattr(vllm_module, "__path__", None)
        if module_path:
            raw_roots = tuple(module_path)
        else:
            module_file = getattr(vllm_module, "__file__", None)
            raw_roots = (Path(module_file).parent,) if module_file else ()
    roots: list[Path] = []
    seen: set[Path] = set()
    for raw_root in raw_roots:
        try:
            root = Path(os.fspath(raw_root)).resolve(strict=True)
        except (OSError, TypeError, ValueError) as error:
            raise UnsupportedRuntimeError("vLLM package root is unreadable") from error
        if not root.is_dir():
            raise UnsupportedRuntimeError("vLLM package root is not a directory")
        if root not in seen:
            roots.append(root)
            seen.add(root)
    if not roots:
        raise UnsupportedRuntimeError("cannot locate a vLLM package root")
    return tuple(roots)


def _package_files(root: Path) -> tuple[Path, ...]:
    try:
        files = tuple(
            sorted(
                (
                    path
                    for path in root.rglob("*")
                    if path.is_file()
                    and not _IGNORED_PACKAGE_PARTS.intersection(
                        path.relative_to(root).parts
                    )
                    and path.suffix not in _IGNORED_PACKAGE_SUFFIXES
                ),
                key=lambda path: path.relative_to(root).as_posix(),
            )
        )
    except OSError as error:
        raise UnsupportedRuntimeError("cannot enumerate vLLM package content") from error
    return files


def _installed_package_sha256(roots: Sequence[Path]) -> str:
    """Hash the bytes actually importable as the installed vLLM package."""

    digest = hashlib.sha256(b"spoolcache:vllm-installed-package:v1\0")
    total_files = 0
    for root_index, root in enumerate(roots):
        files = _package_files(root)
        before_names = tuple(path.relative_to(root).as_posix() for path in files)
        digest.update(root_index.to_bytes(4, "big"))
        for path, relative_name in zip(files, before_names, strict=True):
            try:
                with path.open("rb") as handle:
                    before = os.fstat(handle.fileno())
                    digest.update(relative_name.encode("utf-8"))
                    digest.update(b"\0")
                    digest.update(before.st_size.to_bytes(8, "big"))
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                    after = os.fstat(handle.fileno())
            except OSError as error:
                raise UnsupportedRuntimeError(
                    f"cannot read vLLM package content: {relative_name}"
                ) from error
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise UnsupportedRuntimeError(
                    f"vLLM package content changed while hashing: {relative_name}"
                )
            total_files += 1
        after_names = tuple(
            path.relative_to(root).as_posix() for path in _package_files(root)
        )
        if before_names != after_names:
            raise UnsupportedRuntimeError(
                "vLLM package file set changed while hashing"
            )
    if total_files == 0:
        raise UnsupportedRuntimeError("vLLM package root contains no stable files")
    return digest.hexdigest()


def verify_vllm_runtime(
    *,
    vllm_module: object | None = None,
    connector_base: type | None = None,
    connector_type: type | None = None,
    supports_hma: type | None = None,
    spec_kind_resolver: Callable[[object], object] | object | None = None,
    package_roots: Sequence[Path] | None = None,
    require_pp_aware: bool = False,
) -> RuntimeReceipt:
    """Verify public vLLM call contracts and attest installed runtime bytes."""

    if vllm_module is None:
        vllm_module = import_module("vllm")
    if connector_base is None or supports_hma is None:
        base_module = import_module(
            "vllm.distributed.kv_transfer.kv_connector.v1.base"
        )
        connector_base = getattr(base_module, "KVConnectorBase_V1")
        supports_hma = getattr(base_module, "SupportsHMA")
    if connector_type is None:
        raise UnsupportedRuntimeError(
            "SpoolCache connector type is required for override verification"
        )
    if not isinstance(require_pp_aware, bool):
        raise UnsupportedRuntimeError("PP-aware compatibility request is invalid")
    if inspect.isabstract(connector_type):
        missing = ", ".join(sorted(connector_type.__abstractmethods__))
        raise UnsupportedRuntimeError(
            "SpoolCache connector does not implement vLLM abstract hooks: "
            + missing
        )
    _verify_spec_kind_resolver(spec_kind_resolver)

    detected = getattr(vllm_module, "__version__", None)
    if not isinstance(detected, str) or not detected:
        try:
            detected = version("vllm")
        except PackageNotFoundError as error:
            raise UnsupportedRuntimeError("cannot determine the vLLM version") from error

    constructor = _accepts_required_prefix(
        connector_base.__init__,
        ("self", "vllm_config", "role", "kv_cache_config"),
        label="KVConnectorBase_V1 constructor",
    )
    _assert_override_substitutable(
        connector_base.__init__,
        connector_type.__init__,
        label="KVConnectorBase_V1 constructor",
    )
    hma_parameters = _accepts_required_prefix(
        supports_hma.request_finished_all_groups,
        ("self", "request", "block_ids"),
        label="SupportsHMA completion",
    )
    hma_override = getattr(connector_type, "request_finished_all_groups", None)
    if not callable(hma_override):
        raise UnsupportedRuntimeError(
            "SpoolCache connector is missing request_finished_all_groups"
        )
    _assert_override_substitutable(
        supports_hma.request_finished_all_groups,
        hma_override,
        label="SupportsHMA completion",
    )

    hook_contracts = dict(_CONNECTOR_HOOK_CONTRACTS)
    pp_aware_name = "set_xfer_handshake_metadata_pp_aware"
    base_has_pp_aware = callable(getattr(connector_base, pp_aware_name, None))
    if require_pp_aware and not base_has_pp_aware:
        raise UnsupportedRuntimeError(
            "vLLM connector API lacks the PP-aware handshake hook"
        )
    if base_has_pp_aware:
        hook_contracts.update(_PP_AWARE_HANDSHAKE_CONTRACT)

    missing_base = sorted(
        name
        for name in hook_contracts
        if not callable(getattr(connector_base, name, None))
    )
    missing_override = sorted(
        name
        for name in hook_contracts
        if not callable(getattr(connector_type, name, None))
    )
    if missing_base:
        raise UnsupportedRuntimeError(
            "vLLM connector API is missing hooks: " + ", ".join(missing_base)
        )
    if missing_override:
        raise UnsupportedRuntimeError(
            "SpoolCache connector is missing hooks: " + ", ".join(missing_override)
        )
    for name, (required, variadic_keyword) in hook_contracts.items():
        base_hook = getattr(connector_base, name)
        override_hook = getattr(connector_type, name)
        _accepts_required_prefix(
            base_hook,
            required,
            label=f"KVConnectorBase_V1.{name}",
            variadic_keyword=variadic_keyword,
        )
        _assert_override_substitutable(
            base_hook,
            override_hook,
            label=f"KVConnectorBase_V1.{name}",
        )

    roots = _package_roots(vllm_module, package_roots)
    build_sha256 = _installed_package_sha256(roots)
    return RuntimeReceipt(
        vllm_version=detected,
        vllm_build_sha256=build_sha256,
        build_fingerprint_kind=_FINGERPRINT_KIND,
        compatibility_mode="automatic-contract",
        connector_constructor=constructor,
        supports_hma_parameters=hma_parameters,
        capabilities=tuple(
            sorted((*hook_contracts, "get_kv_cache_spec_kind"))
        ),
    )


def expandable_segments_enabled(environment: Mapping[str, str] | None = None) -> bool:
    """Recognize PyTorch's legacy and current allocator environment names."""

    values = os.environ if environment is None else environment
    for variable in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        for setting in values.get(variable, "").split(","):
            key, separator, raw_value = setting.partition(":")
            if (
                separator
                and key.strip().lower() == "expandable_segments"
                and raw_value.strip().lower() in {"1", "true", "yes", "on"}
            ):
                return True
    return False


def require_qualified_allocator(environment: Mapping[str, str] | None = None) -> None:
    if expandable_segments_enabled(environment):
        raise UnsupportedRuntimeError(
            "SpoolCache GPU access is incompatible with expandable_segments:True "
            "until the target mover is qualified with CUDA virtual memory"
        )


__all__ = [
    "RuntimeReceipt",
    "expandable_segments_enabled",
    "require_qualified_allocator",
    "verify_vllm_runtime",
]
