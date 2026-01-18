"""Custom exceptions for the application."""


class AppException(Exception):
    """Base application exception."""
    
    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ModelNotLoadedException(AppException):
    """Model is not loaded."""
    
    def __init__(self):
        super().__init__("Model is not loaded", "MODEL_NOT_LOADED")


class PredictionException(AppException):
    """Error during prediction."""
    
    def __init__(self, message: str):
        super().__init__(f"Prediction failed: {message}", "PREDICTION_ERROR")


class TrainingException(AppException):
    """Error during training."""
    
    def __init__(self, message: str):
        super().__init__(f"Training failed: {message}", "TRAINING_ERROR")


class StorageException(AppException):
    """Error in storage operations."""
    
    def __init__(self, message: str):
        super().__init__(f"Storage error: {message}", "STORAGE_ERROR")


class ServerConnectionException(AppException):
    """Cannot connect to FL server."""
    
    def __init__(self, server_url: str):
        super().__init__(
            f"Cannot connect to FL server at {server_url}",
            "SERVER_CONNECTION_ERROR"
        )


class ModelDownloadException(AppException):
    """Error downloading model from MinIO."""
    
    def __init__(self, url: str):
        super().__init__(
            f"Failed to download model from {url}",
            "MODEL_DOWNLOAD_ERROR"
        )

