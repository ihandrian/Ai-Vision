"""
Object Detection with YOLOv4-tiny on Video Input
Adapted from window capture to work with video files.
Compatible with trained models from Google Colab.
"""

import numpy as np
import cv2 as cv
import os
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple, Any

from app.logger_config import get_logger
from app.exceptions import VideoProcessingError, StreamConnectionError, ModelLoadError
from app.constants import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CLASSES_FILE,
    DEFAULT_FPS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    YOLO_INPUT_SIZE,
    YOLO_SCALE_FACTOR,
    NMS_THRESHOLD_OFFSET,
    PREDEFINED_COLORS,
    KEY_QUIT,
    KEY_PAUSE,
    STREAM_URL_PREFIXES,
    OUTPUT_VIDEO_SUFFIX,
)

logger = get_logger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    # Get the directory where this file is located
    current_file = Path(__file__).resolve()
    # Go up one level from app/ to project root
    return current_file.parent.parent


class VideoCapture:
    """
    Video capture class to replace WindowCapture.
    Reads frames from video files, webcam, or video streams using OpenCV.
    """
    
    def __init__(self, video_path: Union[str, int, Path]):
        """
        Initialize video capture.
        
        Args:
            video_path: Path to video file, camera index (0 for webcam), or video stream URL
                       Supported URL formats: http://, https://, rtsp://, rtmp://, udp://
        
        Raises:
            VideoProcessingError: If video source cannot be opened
            StreamConnectionError: If stream connection fails
        """
        self.is_camera = False
        self.is_stream = False
        original_video_path = video_path
        
        if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
            # Camera input
            self.cap = cv.VideoCapture(int(video_path))
            self.is_camera = True
            logger.info(f"Initializing camera capture (index {video_path})")
        elif isinstance(video_path, str):
            # Check if it's a URL/stream
            video_path_lower = video_path.lower().strip()
            if video_path_lower.startswith(STREAM_URL_PREFIXES):
                # Video stream URL
                self.cap = cv.VideoCapture(video_path)
                self.is_stream = True
                logger.info(f"Connecting to video stream: {video_path}")
                print(f"Connecting to video stream: {video_path}")
            else:
                # Video file input - resolve relative to project root
                video_path_obj = Path(video_path)
                if not video_path_obj.is_absolute():
                    project_root = get_project_root()
                    video_path_obj = project_root / video_path
                
                if not video_path_obj.exists():
                    error_msg = f'Video file not found: {video_path_obj}'
                    logger.error(error_msg)
                    raise VideoProcessingError(error_msg)
                
                self.cap = cv.VideoCapture(str(video_path_obj))
                logger.info(f"Opening video file: {video_path_obj}")
        else:
            error_msg = f'Invalid video_path type: {type(video_path)}'
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not self.cap.isOpened():
            if self.is_stream:
                error_msg = (
                    f'Could not connect to video stream: {original_video_path}\n'
                    f'Please check the URL and ensure the stream is accessible.'
                )
                logger.error(error_msg)
                raise StreamConnectionError(error_msg)
            else:
                error_msg = f'Could not open video source: {original_video_path}'
                logger.error(error_msg)
                raise VideoProcessingError(error_msg)
        
        # Get video properties
        self.w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        # For streams, frame_count might be 0 or -1
        if self.is_stream:
            fps_str = f"{self.fps:.2f}" if self.fps > 0 else "variable"
            logger.info(f"Stream properties: {self.w}x{self.h}, FPS: {fps_str}")
            print(f"Stream properties: {self.w}x{self.h}, FPS: {fps_str}")
        else:
            logger.info(f"Video properties: {self.w}x{self.h}, FPS: {self.fps:.2f}, Frames: {self.frame_count}")
            print(f"Video properties: {self.w}x{self.h}, FPS: {self.fps:.2f}, Frames: {self.frame_count}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get next frame from video.
        
        Returns:
            numpy array of frame, or None if video ended
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Convert BGR to RGB for consistency (OpenCV uses BGR by default)
        # But we'll keep it as BGR since cv2.imshow expects BGR
        return frame
    
    def get_window_size(self) -> Tuple[int, int]:
        """Get video dimensions."""
        return (self.w, self.h)
    
    def release(self) -> None:
        """Release video capture."""
        self.cap.release()
        logger.debug("Video capture released")
    
    def set_position(self, frame_number: int) -> None:
        """Set video position to specific frame number."""
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        logger.debug(f"Set video position to frame {frame_number}")
    
    def get_current_frame_number(self) -> int:
        """Get current frame number."""
        return int(self.cap.get(cv.CAP_PROP_POS_FRAMES))


class ImageProcessor:
    """
    YOLOv4-tiny image processor for object detection.
    Compatible with models trained in Google Colab.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int],
        cfg_file: Union[str, Path],
        weights_file: Union[str, Path],
        names_file: Optional[Union[str, Path]] = None
    ):
        """
        Initialize YOLOv4-tiny processor.
        
        Args:
            img_size: Tuple of (width, height) for input images
            cfg_file: Path to YOLOv4-tiny config file (.cfg)
            weights_file: Path to trained weights file (.weights)
            names_file: Path to class names file (.names). If None, tries default location.
        
        Raises:
            ModelLoadError: If model files cannot be loaded
            VideoProcessingError: If class names file cannot be found
        """
        np.random.seed(42)
        project_root = get_project_root()
        
        # Resolve paths relative to project root
        cfg_path = Path(cfg_file)
        if not cfg_path.is_absolute():
            cfg_path = project_root / cfg_file
        
        weights_path = Path(weights_file)
        if not weights_path.is_absolute():
            weights_path = project_root / weights_file
        
        # Load network
        if not cfg_path.exists():
            error_msg = f'Config file not found: {cfg_path}'
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
        if not weights_path.exists():
            error_msg = f'Weights file not found: {weights_path}'
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
        
        logger.info(f"Loading model: cfg={cfg_path}, weights={weights_path}")
        try:
            self.net = cv.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
            logger.info("Model loaded successfully")
        except Exception as e:
            error_msg = f'Failed to load model: {e}'
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
        
        # Get output layer names
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i-1] for i in self.net.getUnconnectedOutLayers()]
        
        self.W: int = img_size[0]
        self.H: int = img_size[1]
        
        # Load class names
        names_path: Optional[Path] = None
        if names_file is None:
            # Try default locations relative to project root
            possible_paths = [
                project_root / DEFAULT_CONFIG_DIR / "obj.names",
                project_root / "obj.names",
                cfg_path.parent / "obj.names"
            ]
            for path in possible_paths:
                if path.exists():
                    names_path = path
                    break
        else:
            names_path = Path(names_file)
            if not names_path.is_absolute():
                names_path = project_root / names_file
        
        self.classes: Dict[int, str] = {}
        if names_path and names_path.exists():
            logger.info(f"Loading class names from: {names_path}")
            with open(names_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            for i, line in enumerate(lines):
                self.classes[i] = line.strip()
        else:
            # Fallback: try to load from classes.txt in project root
            classes_file = project_root / DEFAULT_CLASSES_FILE
            if classes_file.exists():
                logger.info(f"Loading class names from fallback: {classes_file}")
                with open(classes_file, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                for i, line in enumerate(lines):
                    self.classes[i] = line.strip()
            else:
                error_msg = 'Could not find class names file. Please provide names_file or ensure classes.txt exists.'
                logger.error(error_msg)
                raise VideoProcessingError(error_msg)
        
        logger.info(f"Loaded {len(self.classes)} classes: {list(self.classes.values())}")
        print(f"Loaded {len(self.classes)} classes: {list(self.classes.values())}")
        
        # Generate colors for classes (supports unlimited classes)
        self.colors: List[Tuple[int, int, int]] = self._generate_colors(len(self.classes))
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each class."""
        if num_classes <= len(PREDEFINED_COLORS):
            # Use predefined colors for small number of classes
            logger.debug(f"Using predefined colors for {num_classes} classes")
            return PREDEFINED_COLORS[:num_classes]
        else:
            # Generate colors for many classes
            logger.debug(f"Generating random colors for {num_classes} classes")
            colors = []
            np.random.seed(42)
            for i in range(num_classes):
                colors.append(tuple(np.random.randint(0, 255, 3).tolist()))
            return colors
    
    def process_image(
        self,
        img: np.ndarray,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        draw: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process image and detect objects.
        
        Args:
            img: Input image as numpy array (BGR format)
            confidence_threshold: Minimum confidence for detections
            draw: Whether to draw bounding boxes on image
        
        Returns:
            List of detected objects with coordinates and class info
        """
        # Create blob from image
        blob = cv.dnn.blobFromImage(
            img,
            YOLO_SCALE_FACTOR,
            YOLO_INPUT_SIZE,
            swapRB=True,
            crop=False
        )
        self.net.setInput(blob)
        
        # Forward pass
        outputs = self.net.forward(self.ln)
        outputs = np.vstack(outputs)
        
        # Get coordinates
        coordinates = self.get_coordinates(outputs, confidence_threshold)
        
        # Draw bounding boxes if requested
        if draw:
            self.draw_identified_objects(img, coordinates)
        
        return coordinates
    
    def get_coordinates(
        self,
        outputs: np.ndarray,
        conf: float
    ) -> List[Dict[str, Any]]:
        """
        Extract bounding box coordinates from YOLO outputs.
        
        Args:
            outputs: YOLO network outputs
            conf: Confidence threshold
        
        Returns:
            List of dictionaries with detection info
        """
        boxes: List[List[int]] = []
        confidences: List[float] = []
        classIDs: List[int] = []
        
        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > conf:
                # Scale coordinates to image size
                x, y, w, h = output[:4] * np.array([self.W, self.H, self.W, self.H])
                p0 = int(x - w//2), int(y - h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)
        
        # Apply Non-Maximum Suppression
        nms_threshold = conf - NMS_THRESHOLD_OFFSET
        indices = cv.dnn.NMSBoxes(boxes, confidences, conf, nms_threshold)
        
        if len(indices) == 0:
            return []
        
        # Format results
        coordinates: List[Dict[str, Any]] = []
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            coordinates.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'class': classIDs[i],
                'class_name': self.classes[classIDs[i]],
                'confidence': confidences[i]
            })
        
        return coordinates
    
    def draw_identified_objects(
        self,
        img: np.ndarray,
        coordinates: List[Dict[str, Any]]
    ) -> None:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            img: Image to draw on (modified in place)
            coordinates: List of detection dictionaries
        """
        for coordinate in coordinates:
            x = coordinate['x']
            y = coordinate['y']
            w = coordinate['w']
            h = coordinate['h']
            classID = coordinate['class']
            confidence = coordinate.get('confidence', 0.0)
            
            # Get color for this class
            color = self.colors[classID % len(self.colors)]
            
            # Draw bounding box
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with confidence
            label = f"{self.classes[classID]} {confidence:.2f}"
            cv.putText(img, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def process_video(
    video_path: Union[str, int, Path],
    cfg_file: Union[str, Path],
    weights_file: Union[str, Path],
    names_file: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    display: bool = True,
    save_output: bool = False
) -> List[Dict[str, Any]]:
    """
    Process video file, webcam, or video stream with YOLOv4-tiny detection.
    
    Args:
        video_path: Path to input video file, camera index (0 for webcam), or video stream URL
                   Supported stream formats: http://, https://, rtsp://, rtmp://, udp://
        cfg_file: Path to YOLOv4-tiny config file
        weights_file: Path to trained weights file
        names_file: Path to class names file (optional)
        output_path: Path to save output video (optional)
        confidence_threshold: Minimum confidence for detections
        display: Whether to display video during processing
        save_output: Whether to save output video
    
    Returns:
        List of all detections (frame_number, detections)
    
    Raises:
        VideoProcessingError: If video processing fails
        ModelLoadError: If model cannot be loaded
    """
    project_root = get_project_root()
    
    # Resolve output path relative to project root if provided
    output_path_obj: Optional[Path] = None
    if output_path:
        output_path_obj = Path(output_path)
        if not output_path_obj.is_absolute():
            output_path_obj = project_root / output_path
    
    logger.info(f"Starting video processing: video_path={video_path}, confidence={confidence_threshold}")
    
    # Initialize video capture
    try:
        video_cap = VideoCapture(video_path)
        img_size = video_cap.get_window_size()
    except (VideoProcessingError, StreamConnectionError) as e:
        logger.error(f"Failed to initialize video capture: {e}")
        raise
    
    # Initialize processor
    try:
        processor = ImageProcessor(img_size, cfg_file, weights_file, names_file)
    except (ModelLoadError, VideoProcessingError) as e:
        logger.error(f"Failed to initialize image processor: {e}")
        video_cap.release()
        raise
    
    # Setup video writer if saving output
    video_writer: Optional[cv.VideoWriter] = None
    if save_output:
        if output_path_obj is None:
            if isinstance(video_path, str):
                base_name = Path(video_path).stem
            else:
                base_name = "output"
            output_path_obj = project_root / f"{base_name}{OUTPUT_VIDEO_SUFFIX}"
        
        # Use default FPS if not available (common for streams)
        fps = video_cap.fps if video_cap.fps > 0 else DEFAULT_FPS
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(
            str(output_path_obj),
            fourcc,
            fps,
            (video_cap.w, video_cap.h)
        )
        if not video_writer.isOpened():
            logger.warning(f"Could not initialize video writer. Output may not be saved.")
            print(f"Warning: Could not initialize video writer. Output may not be saved.")
        else:
            logger.info(f"Saving output to: {output_path_obj} (FPS: {fps:.2f})")
            print(f"Saving output to: {output_path_obj} (FPS: {fps:.2f})")
    
    all_detections: List[Dict[str, Any]] = []
    frame_number = 0
    
    logger.info("Starting video processing loop")
    print("\nProcessing video...")
    print("Press 'q' to quit, 'p' to pause")
    paused = False
    
    try:
        while True:
            if not paused:
                frame = video_cap.get_frame()
                if frame is None:
                    logger.info("End of video reached")
                    break
            
            if frame is not None:
                # Process frame
                detections = processor.process_image(frame, confidence_threshold, draw=True)
                all_detections.append({
                    'frame': frame_number,
                    'detections': detections
                })
                
                # Log detections
                if detections:
                    logger.debug(f"Frame {frame_number}: {len(detections)} detections")
                    print(f"\nFrame {frame_number}:")
                    for det in detections:
                        logger.debug(
                            f"  - {det['class_name']}: confidence={det['confidence']:.2f}, "
                            f"bbox=({det['x']}, {det['y']}, {det['w']}, {det['h']})"
                        )
                        print(f"  - {det['class_name']}: confidence={det['confidence']:.2f}, "
                              f"bbox=({det['x']}, {det['y']}, {det['w']}, {det['h']})")
                
                # Save frame if writing output
                if video_writer:
                    video_writer.write(frame)
                
                # Display frame
                if display:
                    cv.imshow('Object Detection', frame)
            
            # Handle keyboard input
            key = cv.waitKey(1) & 0xFF
            if key == KEY_QUIT:
                logger.info("User requested quit (pressed 'q')")
                break
            elif key == KEY_PAUSE:
                paused = not paused
                logger.debug(f"Video {'paused' if paused else 'resumed'}")
                print("Paused" if paused else "Resumed")
            
            if not paused:
                frame_number += 1
    
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user (Ctrl+C)")
        print("\nProcessing interrupted by user")
    except Exception as e:
        logger.exception(f"Unexpected error during video processing: {e}")
        raise VideoProcessingError(f"Error during video processing: {e}") from e
    
    finally:
        # Cleanup
        video_cap.release()
        if video_writer:
            video_writer.release()
            logger.debug("Video writer released")
        if display:
            cv.destroyAllWindows()
            logger.debug("Display windows closed")
    
    logger.info(f"Finished processing {frame_number} frames. Total detections: {sum(len(d['detections']) for d in all_detections)}")
    print(f'\nFinished processing {frame_number} frames.')
    if save_output and video_writer and output_path_obj:
        logger.info(f"Output saved to: {output_path_obj}")
        print(f'Output saved to: {output_path_obj}')
    
    return all_detections


# This module is designed to be used via main.py
# For interactive detection, please use: python main.py (option 4)
if __name__ == "__main__":
    logger.info("objvision module run directly (should use main.py instead)")
    print("="*60)
    print("  Object Detection Module")
    print("="*60)
    print("\nThis module is designed to be used via main.py")
    print("Please run the main program instead:")
    print("\n  python main.py")
    print("\nThen select option 4: Run object detection on video")
    print("="*60)
    print("\nAlternatively, you can import and use the functions programmatically:")
    print("  from app.objvision import process_video, VideoCapture, ImageProcessor")
    print("="*60)

