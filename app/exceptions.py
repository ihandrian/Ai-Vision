"""
Custom exception classes for the object detection project.
Provides specific exception types for better error handling.
"""


class ObjectDetectionError(Exception):
    """Base exception class for all object detection errors."""
    pass


class VideoProcessingError(ObjectDetectionError):
    """Exception raised for video processing errors."""
    pass


class LabelProcessingError(ObjectDetectionError):
    """Exception raised for label processing errors."""
    pass


class ConfigurationError(ObjectDetectionError):
    """Exception raised for configuration errors."""
    pass


class FileNotFoundError(ObjectDetectionError):
    """Exception raised when a required file is not found."""
    
    def __init__(self, file_path: str, message: str = None):
        """
        Initialize FileNotFoundError.
        
        Args:
            file_path: Path to the file that was not found
            message: Optional custom error message
        """
        self.file_path = file_path
        if message is None:
            message = f"Required file not found: {file_path}"
        super().__init__(message)


class InvalidInputError(ObjectDetectionError):
    """Exception raised for invalid user input."""
    pass


class ModelLoadError(ObjectDetectionError):
    """Exception raised when model loading fails."""
    pass


class StreamConnectionError(VideoProcessingError):
    """Exception raised when video stream connection fails."""
    pass

