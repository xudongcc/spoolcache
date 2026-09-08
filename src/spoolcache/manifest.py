"""Authenticated, rank-local SpoolCache manifest format."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .errors import ManifestError
from .identity import canonical_json

MANIFEST_SCHEMA = "spoolcache-manifest/v1"
ENVELOPE_SCHEMA = "spoolcache-manifest-envelope/v1"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


@dataclass(frozen=True)
class ObjectDescriptor:
    group_index: int
    layer_name: str
    page_start: int
    page_count: int
    byte_length: int
    stored_length: int
    sha256: str
    relative_path: str

    def __post_init__(self) -> None:
        ints = (
            self.group_index,
            self.page_start,
            self.page_count,
            self.byte_length,
            self.stored_length,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in ints):
            raise ManifestError("object descriptor integers are malformed")
        if self.group_index < 0 or self.page_start < 0 or self.page_count <= 0:
            raise ManifestError("object descriptor page range is invalid")
        if self.byte_length <= 0 or self.stored_length < self.byte_length:
            raise ManifestError("object descriptor byte lengths are invalid")
        if not self.layer_name or len(self.layer_name) > 512:
            raise ManifestError("object descriptor layer name is invalid")
        if not _is_sha256(self.sha256):
            raise ManifestError("object descriptor SHA-256 is malformed")
        expected = f"objects/{self.sha256[:2]}/{self.sha256}.spool"
        if self.relative_path != expected:
            raise ManifestError("object path is not derived from its digest")

    @property
    def sort_key(self) -> tuple[int, str, int]:
        return self.group_index, self.layer_name, self.page_start

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObjectDescriptor":
        expected = {
            "group_index",
            "layer_name",
            "page_start",
            "page_count",
            "byte_length",
            "stored_length",
            "sha256",
            "relative_path",
        }
        if set(raw) != expected:
            raise ManifestError("object descriptor fields differ from schema")
        try:
            return cls(**raw)
        except TypeError as error:
            raise ManifestError("object descriptor types are malformed") from error


@dataclass(frozen=True)
class RankManifest:
    entry_id: str
    deployment_identity_digest: str
    rank_identity_digest: str
    span_tokens: int
    physical_rank: int
    topology_digest: str
    profile: str
    layout_digest: str
    objects: tuple[ObjectDescriptor, ...]
    created_at_unix_ns: int
    schema: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        for digest in (
            self.entry_id,
            self.deployment_identity_digest,
            self.rank_identity_digest,
            self.topology_digest,
            self.layout_digest,
        ):
            if not _is_sha256(digest):
                raise ManifestError("manifest contains a malformed SHA-256")
        if self.schema != MANIFEST_SCHEMA:
            raise ManifestError("unsupported manifest schema")
        if (
            isinstance(self.span_tokens, bool)
            or not isinstance(self.span_tokens, int)
            or self.span_tokens <= 0
        ):
            raise ManifestError("manifest span_tokens is invalid")
        if (
            isinstance(self.physical_rank, bool)
            or not isinstance(self.physical_rank, int)
            or self.physical_rank < 0
        ):
            raise ManifestError("manifest physical_rank is invalid")
        if (
            isinstance(self.created_at_unix_ns, bool)
            or not isinstance(self.created_at_unix_ns, int)
            or self.created_at_unix_ns <= 0
        ):
            raise ManifestError("manifest creation timestamp is invalid")
        if not self.profile or len(self.profile) > 128:
            raise ManifestError("manifest profile is invalid")
        if not self.objects:
            raise ManifestError("manifest has no objects")
        keys = [descriptor.sort_key for descriptor in self.objects]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ManifestError("manifest objects must be sorted and unique")

    @property
    def logical_bytes(self) -> int:
        return sum(item.byte_length for item in self.objects)

    @property
    def stored_bytes(self) -> int:
        return sum(item.stored_length for item in self.objects)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objects"] = [asdict(item) for item in self.objects]
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> "RankManifest":
        expected = {
            "entry_id",
            "deployment_identity_digest",
            "rank_identity_digest",
            "span_tokens",
            "physical_rank",
            "topology_digest",
            "profile",
            "layout_digest",
            "objects",
            "created_at_unix_ns",
            "schema",
        }
        if set(raw) != expected:
            raise ManifestError("manifest fields differ from schema")
        objects = raw.get("objects")
        if not isinstance(objects, list):
            raise ManifestError("manifest objects must be a list")
        try:
            values = dict(raw)
            values["objects"] = tuple(ObjectDescriptor.from_dict(item) for item in objects)
            return cls(**values)
        except (TypeError, AttributeError) as error:
            raise ManifestError("manifest payload types are malformed") from error


@dataclass(frozen=True)
class ManifestEnvelope:
    manifest: RankManifest
    payload_sha256: str

    @property
    def manifest_digest(self) -> str:
        return hashlib.sha256(encode_manifest(self.manifest)).hexdigest()


def encode_manifest(manifest: RankManifest) -> bytes:
    payload = manifest.to_payload()
    payload_bytes = canonical_json(payload)
    envelope = {
        "schema": ENVELOPE_SCHEMA,
        "payload": payload,
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }
    encoded = canonical_json(envelope) + b"\n"
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ManifestError("manifest exceeds the size limit")
    return encoded


def decode_manifest(encoded: bytes | bytearray | memoryview) -> ManifestEnvelope:
    raw_bytes = bytes(encoded)
    if not raw_bytes or len(raw_bytes) > MAX_MANIFEST_BYTES:
        raise ManifestError("manifest length is invalid")
    try:
        envelope = json.loads(raw_bytes)
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ManifestError("manifest JSON is invalid") from error
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema",
        "payload",
        "payload_sha256",
    }:
        raise ManifestError("manifest envelope fields differ from schema")
    if envelope["schema"] != ENVELOPE_SCHEMA:
        raise ManifestError("manifest envelope schema is unsupported")
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise ManifestError("manifest payload must be an object")
    expected = envelope["payload_sha256"]
    if not _is_sha256(expected):
        raise ManifestError("manifest payload digest is malformed")
    try:
        actual = hashlib.sha256(canonical_json(payload)).hexdigest()
    except ManifestError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        # JSON accepts non-standard values such as NaN by default, and deeply
        # nested or resource-limited inputs can fail only during canonical
        # re-encoding. These remain untrusted manifest data failures rather
        # than escaping as startup-fatal implementation exceptions.
        raise ManifestError("manifest payload is not canonical JSON") from error
    if actual != expected:
        raise ManifestError("manifest payload digest differs")
    try:
        manifest = RankManifest.from_payload(payload)
    except ManifestError:
        raise
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ManifestError("manifest payload is invalid") from error
    return ManifestEnvelope(manifest=manifest, payload_sha256=actual)


def ordered_descriptors(
    descriptors: Sequence[ObjectDescriptor],
) -> tuple[ObjectDescriptor, ...]:
    return tuple(sorted(descriptors, key=lambda item: item.sort_key))
