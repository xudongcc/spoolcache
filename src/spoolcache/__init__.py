"""SpoolCache persistent KV cache primitives."""

from .config import SpoolCacheConfig
from .hma import (
    HMALayout,
    build_hma_layout,
)
from .identity import DeploymentIdentity, RankIdentity
from .manifest import PageSlice, TokenFileDescriptor, TokenSnapshot
from .prefix import (
    MultimodalFeatureIdentity,
    PrefixDigest,
    aligned_prefix_span,
    prefix_digests,
    validate_multimodal_features,
)
from .token_files import TokenFileStore

__all__ = [
    "DeploymentIdentity",
    "HMALayout",
    "TokenFileStore",
    "TokenFileDescriptor",
    "PageSlice",
    "PrefixDigest",
    "MultimodalFeatureIdentity",
    "RankIdentity",
    "TokenSnapshot",
    "SpoolCacheConfig",
    "aligned_prefix_span",
    "build_hma_layout",
    "prefix_digests",
    "validate_multimodal_features",
]

__version__ = "0.2.0"
