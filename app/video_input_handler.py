"""
Video input handler module.
Centralizes video input selection logic to reduce code duplication.
"""

import os
from pathlib import Path
from typing import Union, Optional, Tuple
from app.video_utils import find_video_files
from app.exceptions import InvalidInputError, VideoProcessingError
from app.constants import (
    VIDEO_INPUT_MEDIA_FOLDER,
    VIDEO_INPUT_CUSTOM_PATH,
    VIDEO_INPUT_WEBCAM,
    VIDEO_INPUT_STREAM,
    DEFAULT_VIDEO_INPUT_CHOICE,
    DEFAULT_CAMERA_INDEX,
    STREAM_URL_PREFIXES,
    DEFAULT_YES,
)
from app.logger_config import get_logger

logger = get_logger(__name__)


class VideoInputHandler:
    """Handles video input selection from various sources."""
    
    def __init__(self, media_dir: Union[str, Path] = "media"):
        """
        Initialize video input handler.
        
        Args:
            media_dir: Directory containing video files
        """
        self.media_dir = Path(media_dir)
        logger.debug(f"Initialized VideoInputHandler with media_dir: {media_dir}")
    
    def display_video_input_options(self) -> None:
        """Display available video input options to the user."""
        print("\nVideo Input Options:")
        print("  1. Select from media folder")
        print("  2. Enter custom video file path")
        print("  3. Use webcam/camera")
        print("  4. Video stream URL (http/https/rtsp/rtmp)")
    
    def get_video_input(
        self,
        choice: Optional[str] = None,
        require_confirmation: bool = False
    ) -> Tuple[Union[str, int], str]:
        """
        Get video input from user based on their choice.
        
        Args:
            choice: User's choice (1-4). If None, prompts user.
            require_confirmation: Whether to require confirmation for webcam/stream
        
        Returns:
            Tuple of (video_path, input_type) where:
            - video_path: Path to video file, camera index (int), or stream URL
            - input_type: One of 'file', 'camera', 'stream'
        
        Raises:
            InvalidInputError: If user input is invalid
            VideoProcessingError: If video source cannot be accessed
        """
        if choice is None:
            choice = input("\nSelect option (1-4, default 1): ").strip() or DEFAULT_VIDEO_INPUT_CHOICE
        else:
            choice = choice.strip() or DEFAULT_VIDEO_INPUT_CHOICE
        
        logger.info(f"User selected video input option: {choice}")
        
        if choice == VIDEO_INPUT_MEDIA_FOLDER:
            return self._select_from_media_folder()
        elif choice == VIDEO_INPUT_CUSTOM_PATH:
            return self._get_custom_path()
        elif choice == VIDEO_INPUT_WEBCAM:
            return self._get_webcam_input(require_confirmation)
        elif choice == VIDEO_INPUT_STREAM:
            return self._get_stream_input(require_confirmation)
        else:
            raise InvalidInputError(f"Invalid option: {choice}. Please select 1-4.")
    
    def _select_from_media_folder(self) -> Tuple[str, str]:
        """Select video from media folder."""
        video_files = find_video_files(str(self.media_dir))
        
        if not video_files:
            error_msg = (
                f"No video files found in {self.media_dir} folder! "
                "Please place video files in the media folder or use option 2 to specify a custom path."
            )
            logger.warning(error_msg)
            raise VideoProcessingError(error_msg)
        
        print("\nAvailable video files:")
        for i, video in enumerate(video_files, 1):
            print(f"  {i}. {os.path.basename(video)}")
        
        try:
            video_choice = input(f"\nSelect video (1-{len(video_files)}) or press Enter for first: ").strip()
            if not video_choice:
                video_choice = "1"
            
            video_index = int(video_choice) - 1
            if 0 <= video_index < len(video_files):
                selected_video = video_files[video_index]
                logger.info(f"Selected video: {selected_video}")
                return selected_video, "file"
            else:
                raise InvalidInputError(f"Invalid selection: {video_choice}. Please select 1-{len(video_files)}.")
        except ValueError as e:
            raise InvalidInputError(f"Invalid input: {e}")
    
    def _get_custom_path(self) -> Tuple[str, str]:
        """Get custom video file path from user."""
        custom_path = input("\nEnter video file path: ").strip()
        if not custom_path:
            raise InvalidInputError("No path provided!")
        
        # Resolve relative to current working directory if not absolute
        if not os.path.isabs(custom_path):
            custom_path = os.path.join(os.getcwd(), custom_path)
        
        custom_path = Path(custom_path).resolve()
        
        if not custom_path.exists():
            error_msg = f"Video file not found: {custom_path}"
            logger.error(error_msg)
            raise VideoProcessingError(error_msg)
        
        logger.info(f"Using custom video path: {custom_path}")
        return str(custom_path), "file"
    
    def _get_webcam_input(self, require_confirmation: bool) -> Tuple[int, str]:
        """Get webcam/camera input from user."""
        if require_confirmation:
            print("\n⚠ Warning: Extracting frames from webcam will capture frames continuously.")
            print("This is useful for live labeling but may generate many images.")
            confirm = input("Continue with webcam? (y/n): ").strip().lower()
            if confirm != DEFAULT_YES:
                logger.info("User cancelled webcam input")
                raise InvalidInputError("Webcam input cancelled by user")
        
        camera_input = input(
            "\nEnter camera index (0 for default webcam, or press Enter for 0): "
        ).strip()
        
        if camera_input:
            try:
                camera_index = int(camera_input)
                if camera_index < 0:
                    raise ValueError("Camera index must be non-negative")
                logger.info(f"Using camera index: {camera_index}")
                return camera_index, "camera"
            except ValueError as e:
                logger.warning(f"Invalid camera index '{camera_input}', using default (0): {e}")
                return DEFAULT_CAMERA_INDEX, "camera"
        else:
            logger.info("Using default camera index: 0")
            return DEFAULT_CAMERA_INDEX, "camera"
    
    def _get_stream_input(self, require_confirmation: bool) -> Tuple[str, str]:
        """Get video stream URL from user."""
        if require_confirmation:
            print("\n⚠ Note: Extracting frames from video stream will download and process the stream.")
            print("This may take time depending on stream quality and length.")
        
        print("\nSupported stream formats:")
        print("  - HTTP/HTTPS: http://example.com/video.mp4")
        print("  - RTSP: rtsp://username:password@ip:port/stream")
        print("  - RTMP: rtmp://server/live/stream")
        print("  - UDP: udp://@:port")
        
        stream_url = input("\nEnter video stream URL: ").strip()
        if not stream_url:
            raise InvalidInputError("No URL provided!")
        
        # Basic URL validation
        stream_url_lower = stream_url.lower()
        if not stream_url_lower.startswith(STREAM_URL_PREFIXES):
            warning_msg = (
                "Warning: URL doesn't start with http://, https://, rtsp://, rtmp://, or udp://"
            )
            logger.warning(warning_msg)
            print(warning_msg)
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm != DEFAULT_YES:
                logger.info("User cancelled stream input due to invalid URL format")
                raise InvalidInputError("Stream input cancelled by user")
        
        logger.info(f"Using video stream URL: {stream_url}")
        return stream_url, "stream"
    
    def is_camera(self, video_path: Union[str, int]) -> bool:
        """Check if video path is a camera index."""
        return isinstance(video_path, int)
    
    def is_stream(self, video_path: Union[str, int]) -> bool:
        """Check if video path is a stream URL."""
        if isinstance(video_path, str):
            return video_path.lower().strip().startswith(STREAM_URL_PREFIXES)
        return False

