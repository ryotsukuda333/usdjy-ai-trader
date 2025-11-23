"""Custom exception classes for USDJPY Trader system."""


class TraderError(Exception):
    """Base exception class for all trader-related errors.

    Attributes:
        error_code: Machine-readable error identifier
        user_message: User-friendly error message for display
        technical_message: Detailed technical message for logging/debugging
    """

    def __init__(self, error_code: str, user_message: str, technical_message: str):
        """Initialize TraderError with error details.

        Args:
            error_code: Machine-readable error identifier (e.g., 'DATA_FETCH_FAILED')
            user_message: User-friendly message explaining what went wrong
            technical_message: Technical details for debugging (e.g., stack trace, API response)
        """
        self.error_code = error_code
        self.user_message = user_message
        self.technical_message = technical_message
        super().__init__(user_message)


class DataError(TraderError):
    """Exception for data fetch, validation, and loading errors."""
    pass


class FeatureEngineeringError(TraderError):
    """Exception for feature generation and calculation errors."""
    pass


class ModelError(TraderError):
    """Exception for model training, loading, and prediction errors."""
    pass


class BacktestError(TraderError):
    """Exception for backtest execution and logic errors."""
    pass


class VisualizationError(TraderError):
    """Exception for plot generation and visualization errors."""
    pass
