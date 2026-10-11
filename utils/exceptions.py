"""Domain exceptions with business-readable messages (shown in the UI as-is)."""


class CustomerIntelligenceError(Exception):
    """Base error. Message must be actionable, business-readable English."""


class ConfigurationError(CustomerIntelligenceError):
    """Invalid config.yaml values or missing required configuration."""


class DataIngestionError(CustomerIntelligenceError):
    """Upload cannot be read: empty file, unsupported format, undecodable content."""


class MappingError(CustomerIntelligenceError):
    """Column mapping is incomplete or ambiguous (e.g. required field unmapped)."""


class CleaningError(CustomerIntelligenceError):
    """Cleaning configuration is invalid or cleaning produced an empty dataset."""


class InsufficientDataError(CustomerIntelligenceError):
    """Not enough valid rows/customers for the requested computation."""


class ChurnLeakageError(CustomerIntelligenceError):
    """Post-cutoff information used as a predictor, or outcome window missing from data."""
