"""SpoolCache exception hierarchy."""


class SpoolCacheError(RuntimeError):
    """Base class for errors raised by SpoolCache."""


class ConfigurationError(SpoolCacheError, ValueError):
    """Configuration is invalid or unsafe."""


class IdentityError(SpoolCacheError, ValueError):
    """A deployment or rank identity is malformed."""


class ManifestError(SpoolCacheError, ValueError):
    """A manifest is malformed, incompatible, or corrupt."""


class LayoutError(SpoolCacheError, ValueError):
    """The runtime HMA layout differs from the selected strict profile."""


class ObjectCorruptionError(ManifestError):
    """An immutable object's size or digest is invalid."""


class StoreBusyError(SpoolCacheError):
    """A bounded resource has no immediately available credit."""


class UnsupportedRuntimeError(SpoolCacheError):
    """The active vLLM/model runtime is not qualified."""


class FatalRestoreError(SpoolCacheError):
    """A post-admission restore failed and the engine must stop."""
