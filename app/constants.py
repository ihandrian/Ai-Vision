"""
Constants and configuration values for the object detection project.
Centralizes magic numbers and default values.
"""

from pathlib import Path
from typing import List, Tuple

# Default paths
DEFAULT_MEDIA_DIR = Path("media")
DEFAULT_IMAGES_DIR = DEFAULT_MEDIA_DIR / "images"
DEFAULT_OBJ_DIR = DEFAULT_MEDIA_DIR / "obj"
DEFAULT_CLASSES_FILE = Path("classes.txt")
DEFAULT_CONFIG_DIR = Path("yolov4-tiny")
DEFAULT_WEIGHTS_FILE = Path("yolov4-tiny-custom_last.weights")
DEFAULT_CFG_FILE = DEFAULT_CONFIG_DIR / "yolov4-tiny-custom.cfg"
DEFAULT_NAMES_FILE = DEFAULT_CONFIG_DIR / "obj.names"

# Video processing constants
DEFAULT_FRAME_INTERVAL: int = 30
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
DEFAULT_FPS: float = 30.0
PROGRESS_UPDATE_INTERVAL: int = 10  # Update progress every N frames

# YOLO model constants
YOLO_INPUT_SIZE: Tuple[int, int] = (416, 416)
YOLO_SCALE_FACTOR: float = 1.0 / 255.0
NMS_THRESHOLD_OFFSET: float = 0.1  # NMS threshold = confidence - offset

# YOLOv4 configuration calculation constants
MIN_BATCHES: int = 6000
BATCHES_PER_CLASS: int = 2000
FILTERS_PER_ANCHOR: int = 3
ANCHOR_BOXES: int = 3
YOLO_PARAMS_PER_DETECTION: int = 5  # x, y, w, h, confidence

# Video file extensions
SUPPORTED_VIDEO_EXTENSIONS: List[str] = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

# Image file extensions
SUPPORTED_IMAGE_EXTENSIONS: List[str] = ['.jpg', '.jpeg', '.png']

# Stream URL prefixes
STREAM_URL_PREFIXES: Tuple[str, ...] = (
    'http://', 'https://', 'rtsp://', 'rtmp://', 'udp://'
)

# Color constants (BGR format for OpenCV)
COLOR_RED: Tuple[int, int, int] = (0, 0, 255)
COLOR_GREEN: Tuple[int, int, int] = (0, 255, 0)
COLOR_BLUE: Tuple[int, int, int] = (255, 0, 0)
COLOR_CYAN: Tuple[int, int, int] = (255, 255, 0)
COLOR_MAGENTA: Tuple[int, int, int] = (255, 0, 255)
COLOR_YELLOW: Tuple[int, int, int] = (0, 255, 255)

# Predefined colors for small number of classes
PREDEFINED_COLORS: List[Tuple[int, int, int]] = [
    COLOR_RED,
    COLOR_GREEN,
    COLOR_BLUE,
    COLOR_CYAN,
    COLOR_MAGENTA,
    COLOR_YELLOW
]

# Keyboard controls
KEY_QUIT: int = ord('q')
KEY_PAUSE: int = ord('p')
KEY_ESC: int = 27

# Menu options
MENU_EXTRACT_FRAMES: str = "1"
MENU_PROCESS_LABELS: str = "2"
MENU_VIEW_CLASSES: str = "3"
MENU_RUN_DETECTION: str = "4"
MENU_EXIT: str = "5"

# Video input options
VIDEO_INPUT_MEDIA_FOLDER: str = "1"
VIDEO_INPUT_CUSTOM_PATH: str = "2"
VIDEO_INPUT_WEBCAM: str = "3"
VIDEO_INPUT_STREAM: str = "4"

# User input defaults
DEFAULT_VIDEO_INPUT_CHOICE: str = VIDEO_INPUT_MEDIA_FOLDER
DEFAULT_CAMERA_INDEX: int = 0
DEFAULT_YES: str = "y"
DEFAULT_NO: str = "n"

# Makesense.ai related
MAKESENSE_URL: str = "https://www.makesense.ai/"
CLASSES_TXT_FILENAME: str = "classes.txt"

# File naming
IMAGE_FILENAME_PATTERN: str = "img_{:05d}.jpg"
OUTPUT_VIDEO_SUFFIX: str = "_detected.mp4"
CAMERA_OUTPUT_PREFIX: str = "camera_{}_output"
STREAM_OUTPUT_PREFIX: str = "stream_{}"

# Calculation functions
def calculate_filters(num_classes: int) -> int:
    """
    Calculate number of filters for YOLOv4 based on number of classes.
    
    Formula: (num_classes + 5) * 3
    - 5 parameters: x, y, w, h, confidence
    - 3 anchor boxes per grid cell
    
    Args:
        num_classes: Number of object classes
    
    Returns:
        Number of filters needed
    """
    return (num_classes + YOLO_PARAMS_PER_DETECTION) * FILTERS_PER_ANCHOR


def calculate_max_batches(num_classes: int) -> int:
    """
    Calculate maximum batches for YOLOv4 training.
    
    Formula: max(6000, num_classes * 2000)
    
    Args:
        num_classes: Number of object classes
    
    Returns:
        Maximum number of training batches
    """
    return max(MIN_BATCHES, num_classes * BATCHES_PER_CLASS)

