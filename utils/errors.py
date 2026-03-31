"""
Custom exception classes for EEG Analysis Platform.

Provides specific error types for better error handling and debugging.
"""


class EEGAnalysisError(Exception):
    """Base exception for all EEG analysis errors."""

    pass


class InvalidDataError(EEGAnalysisError):
    """Raised when input data format is invalid."""

    pass


class FilterError(EEGAnalysisError):
    """Raised when filter application fails."""

    pass


class ICAError(EEGAnalysisError):
    """Raised when ICA computation fails."""

    pass


class ModelError(EEGAnalysisError):
    """Raised when model operation fails."""

    pass


class APIError(EEGAnalysisError):
    """Raised when external API call fails."""

    pass


class CacheError(EEGAnalysisError):
    """Raised when caching operation fails."""

    pass
