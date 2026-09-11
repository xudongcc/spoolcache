"""Stable exact-prefix digests shared by scheduler and workers."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Iterable, Sequence

from .errors import IdentityError

_TOKEN = struct.Struct("<q")
_COUNT = struct.Struct("<Q")
# Removing the operator namespace changes the encoding; keep legacy keys disjoint.
_DOMAIN = b"spoolcache-exact-prefix/v2\x00"
_MULTIMODAL_DOMAIN = b"spoolcache-multimodal-prefix/v1\x00"
_MULTIMODAL_GEOMETRY = struct.Struct("<QQ")


@dataclass(frozen=True)
class MultimodalFeatureIdentity:
    """Content identity and prompt geometry for one encoder input."""

    modality: str
    identifier: str
    offset: int
    length: int


@dataclass(frozen=True)
class PrefixDigest:
    span_tokens: int
    digest: str


def validate_multimodal_features(
    features: Sequence[MultimodalFeatureIdentity],
    token_count: int,
) -> tuple[MultimodalFeatureIdentity, ...]:
    """Canonicalize complete vLLM media identities or fail closed."""

    if isinstance(token_count, bool) or not isinstance(token_count, int):
        raise ValueError("multimodal token count must be an integer")
    if token_count < 0:
        raise ValueError("multimodal token count must be non-negative")
    try:
        normalized = tuple(features)
    except TypeError as error:
        raise ValueError("multimodal features must be a sequence") from error
    previous_end = 0
    for feature in normalized:
        if not isinstance(feature, MultimodalFeatureIdentity):
            raise ValueError("multimodal feature identity has an invalid type")
        if not isinstance(feature.modality, str) or not feature.modality:
            raise ValueError("multimodal feature modality must be non-empty")
        if not isinstance(feature.identifier, str) or not feature.identifier:
            raise ValueError("multimodal feature identifier must be non-empty")
        if (
            isinstance(feature.offset, bool)
            or not isinstance(feature.offset, int)
            or feature.offset < 0
            or isinstance(feature.length, bool)
            or not isinstance(feature.length, int)
            or feature.length <= 0
        ):
            raise ValueError("multimodal feature geometry is invalid")
        end = feature.offset + feature.length
        if feature.offset < previous_end:
            raise ValueError("multimodal feature ranges must be ordered and disjoint")
        if end > token_count:
            raise ValueError("multimodal feature range exceeds the prompt")
        try:
            feature.modality.encode("utf-8")
            feature.identifier.encode("utf-8")
        except UnicodeError as error:
            raise ValueError("multimodal feature identity is not UTF-8") from error
        previous_end = end
    return normalized


def _multimodal_payload(
    features: Sequence[MultimodalFeatureIdentity], span_tokens: int
) -> bytes:
    active = tuple(feature for feature in features if feature.offset < span_tokens)
    if not active:
        return b""
    encoded = bytearray(_MULTIMODAL_DOMAIN + _COUNT.pack(len(active)))
    for feature in active:
        modality = feature.modality.encode("utf-8")
        identifier = feature.identifier.encode("utf-8")
        encoded.extend(_MULTIMODAL_GEOMETRY.pack(feature.offset, feature.length))
        encoded.extend(_COUNT.pack(len(modality)))
        encoded.extend(modality)
        encoded.extend(_COUNT.pack(len(identifier)))
        encoded.extend(identifier)
    return bytes(encoded)


def aligned_prefix_span(
    prompt_tokens: int,
    *,
    alignment: int,
    chunk_tokens: int,
    min_span_tokens: int = 0,
    max_span_tokens: int | None = None,
) -> int:
    """Return the safe external-KV boundary, leaving one token to compute."""

    values = (prompt_tokens, alignment, chunk_tokens, min_span_tokens)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("prefix alignment values must be integers")
    if prompt_tokens <= 1 or alignment <= 0 or chunk_tokens <= 0:
        return 0
    quantum = alignment * chunk_tokens // _gcd(alignment, chunk_tokens)
    span = ((prompt_tokens - 1) // quantum) * quantum
    if max_span_tokens is not None:
        if isinstance(max_span_tokens, bool) or not isinstance(max_span_tokens, int):
            raise ValueError("max_span_tokens must be an integer")
        span = min(span, (max_span_tokens // quantum) * quantum)
    return span if span >= min_span_tokens else 0


def _gcd(left: int, right: int) -> int:
    while right:
        left, right = right, left % right
    return left


def _initial_digest(deployment_digest: str, cache_salt: str) -> bytes:
    if len(deployment_digest) != 64 or any(
        character not in "0123456789abcdef" for character in deployment_digest
    ):
        raise IdentityError("deployment digest is not SHA-256")
    try:
        deployment = bytes.fromhex(deployment_digest)
    except ValueError as error:
        raise IdentityError("deployment digest is not hexadecimal") from error
    salt_bytes = cache_salt.encode("utf-8")
    return hashlib.sha256(
        _DOMAIN
        + deployment
        + _COUNT.pack(len(salt_bytes))
        + salt_bytes
    ).digest()


def prefix_digests(
    token_ids: Sequence[int] | Iterable[int],
    *,
    deployment_digest: str,
    cache_salt: str = "",
    chunk_tokens: int,
    boundaries: Iterable[int] | None = None,
    multimodal_features: Sequence[MultimodalFeatureIdentity] = (),
) -> tuple[PrefixDigest, ...]:
    """Hash token chunks plus media identity at each requested boundary."""

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    # vLLM supplies a list-like token buffer.  Do not duplicate that million-
    # token container merely to hash it; materialize only genuinely streaming
    # iterables whose length is otherwise unknowable.
    tokens = token_ids if isinstance(token_ids, Sequence) else tuple(token_ids)
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError("token IDs must be integers")
        if not -(1 << 63) <= token < (1 << 63):
            raise ValueError("token ID is outside signed 64-bit range")
    requested = (
        set(range(chunk_tokens, len(tokens) + 1, chunk_tokens))
        if boundaries is None
        else set(boundaries)
    )
    if any(
        isinstance(boundary, bool)
        or not isinstance(boundary, int)
        or boundary <= 0
        or boundary > len(tokens)
        or boundary % chunk_tokens
        for boundary in requested
    ):
        raise ValueError("prefix boundaries must be positive aligned token counts")
    features = validate_multimodal_features(multimodal_features, len(tokens))
    previous = _initial_digest(deployment_digest, cache_salt)
    result: list[PrefixDigest] = []
    for start in range(0, len(tokens), chunk_tokens):
        chunk = tokens[start : start + chunk_tokens]
        if len(chunk) < chunk_tokens:
            break
        encoded = bytearray(_COUNT.pack(len(chunk)))
        for token in chunk:
            encoded.extend(_TOKEN.pack(token))
        previous = hashlib.sha256(previous + encoded).digest()
        span = start + chunk_tokens
        if span in requested:
            media = _multimodal_payload(features, span)
            digest = hashlib.sha256(previous + media).hexdigest() if media else previous.hex()
            result.append(PrefixDigest(span, digest))
    return tuple(result)
