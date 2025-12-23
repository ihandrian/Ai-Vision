import cv2
import os
from pathlib import Path
from typing import Union, Optional
from app.logger_config import get_logger
from app.exceptions import VideoProcessingError, StreamConnectionError
from app.constants import (
    DEFAULT_IMAGES_DIR,
    DEFAULT_FRAME_INTERVAL,
    PROGRESS_UPDATE_INTERVAL,
    IMAGE_FILENAME_PATTERN,
    KEY_ESC,
    STREAM_URL_PREFIXES,
)

logger = get_logger(__name__)


class VideoFrameExtractor:
    """Extract frames from video files for dataset preparation."""
    
    def __init__(
        self,
        video_path: Union[str, int],
        output_dir: Union[str, Path] = DEFAULT_IMAGES_DIR,
        frame_interval: int = DEFAULT_FRAME_INTERVAL
    ):
        """
        Initialize video frame extractor.
        
        Args:
            video_path: Path to video file, camera index (0 for webcam), or video stream URL
                       Supported URL formats: http://, https://, rtsp://, rtmp://, udp://
            output_dir: Directory to save extracted frames
            frame_interval: Extract every Nth frame (default: 30, meaning 1 frame per second for 30fps video)
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized VideoFrameExtractor: output_dir={self.output_dir}, interval={frame_interval}")
    
    def extract_frames(self, max_frames: Optional[int] = None) -> int:
        """
        Extract frames from video and save to output directory.
        
        Args:
            max_frames: Maximum number of frames to extract (None for all frames, useful for webcam/stream)
        
        Returns:
            Number of frames extracted
        """
        # Handle different video source types
        is_camera = isinstance(self.video_path, int) or (
            isinstance(self.video_path, str) and self.video_path.isdigit()
        )
        is_stream = (
            isinstance(self.video_path, str) and
            self.video_path.lower().strip().startswith(STREAM_URL_PREFIXES)
        )
        
        if is_camera:
            # Camera input
            cap = cv2.VideoCapture(int(self.video_path))
            if max_frames is None:
                logger.warning("No frame limit set for webcam. Press Ctrl+C to stop extraction.")
                print("⚠ Warning: No frame limit set for webcam. Press Ctrl+C to stop extraction.")
                print("Extracting frames continuously...")
        elif is_stream:
            # Stream URL
            cap = cv2.VideoCapture(self.video_path)
            if max_frames is None:
                logger.warning("No frame limit set for stream. Will extract until stream ends.")
                print("⚠ Warning: No frame limit set for stream. Will extract until stream ends.")
        else:
            # Video file input
            video_path = Path(self.video_path)
            if not video_path.exists():
                error_msg = f"Video file not found: {self.video_path}"
                logger.error(error_msg)
                raise VideoProcessingError(error_msg)
            cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            if is_stream:
                error_msg = f"Could not connect to video stream: {self.video_path}"
                logger.error(error_msg)
                raise StreamConnectionError(error_msg)
            else:
                error_msg = f"Could not open video source: {self.video_path}"
                logger.error(error_msg)
                raise VideoProcessingError(error_msg)
        
        frame_count = 0
        saved_count = 0
        
        if is_camera:
            logger.info(f"Extracting frames from camera (index {self.video_path})...")
            print(f"Extracting frames from camera (index {self.video_path})...")
        elif is_stream:
            logger.info(f"Extracting frames from stream: {self.video_path}")
            print(f"Extracting frames from stream: {self.video_path}")
        else:
            logger.info(f"Extracting frames from: {self.video_path}")
            print(f"Extracting frames from: {self.video_path}")
        
        logger.info(f"Saving to: {self.output_dir}, interval: {self.frame_interval}")
        print(f"Saving to: {self.output_dir}")
        print(f"Frame interval: Every {self.frame_interval} frames")
        if max_frames:
            logger.info(f"Maximum frames to extract: {max_frames}")
            print(f"Maximum frames to extract: {max_frames}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame at specified interval
                if frame_count % self.frame_interval == 0:
                    frame_filename = self.output_dir / IMAGE_FILENAME_PATTERN.format(saved_count)
                    cv2.imwrite(str(frame_filename), frame)
                    saved_count += 1
                    
                    if saved_count % PROGRESS_UPDATE_INTERVAL == 0:
                        logger.debug(f"Extracted {saved_count} frames...")
                        print(f"Extracted {saved_count} frames...")
                    
                    # Check max_frames limit
                    if max_frames and saved_count >= max_frames:
                        logger.info(f"Reached maximum frame limit ({max_frames})")
                        print(f"\nReached maximum frame limit ({max_frames})")
                        break
                
                frame_count += 1
        except KeyboardInterrupt:
            logger.warning("Extraction interrupted by user (Ctrl+C)")
            print("\n\nExtraction interrupted by user (Ctrl+C)")
        
        cap.release()
        logger.info(f"Extraction complete! Total frames extracted: {saved_count}, processed: {frame_count}")
        print(f"\nExtraction complete! Total frames extracted: {saved_count}")
        print(f"Total video frames processed: {frame_count}")
        return saved_count
    
    def play_video(self, window_name: str = "Video Player") -> None:
        """Play video file in a window."""
        video_path = Path(self.video_path)
        if not video_path.exists():
            error_msg = f"Video file not found: {self.video_path}"
            logger.error(error_msg)
            raise VideoProcessingError(error_msg)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            error_msg = f"Could not open video file: {self.video_path}"
            logger.error(error_msg)
            raise VideoProcessingError(error_msg)
        
        logger.info(f"Playing video: {self.video_path}")
        print(f"Playing video: {self.video_path}")
        print("Press ESC to exit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Video playback ended.")
        print("Video playback ended.")


def find_video_files(media_dir: Union[str, Path] = "media") -> list[str]:
    """Find all video files in the media directory."""
    from app.constants import SUPPORTED_VIDEO_EXTENSIONS
    
    media_path = Path(media_dir)
    video_files: list[str] = []
    
    if not media_path.exists():
        logger.warning(f"Media directory not found: {media_dir}")
        return video_files
    
    for file in media_path.iterdir():
        if file.is_file() and any(file.suffix.lower() == ext for ext in SUPPORTED_VIDEO_EXTENSIONS):
            video_files.append(str(file))
    
    logger.debug(f"Found {len(video_files)} video files in {media_dir}")
    return video_files

