# Object Detection Dataset Generator

A comprehensive Python-based tool for generating and preparing custom object detection datasets from videos, specifically designed for YOLOv4-tiny training with makesense.ai integration.

## Overview

This project provides a complete workflow to:
- Extract frames from video files and prepare for makesense.ai
- Process labels from makesense.ai
- Prepare training data for YOLOv4-tiny
- Create training-ready datasets

## Features

- **Video Frame Extraction**: Extract frames from video files with configurable intervals
- **Makesense.ai Integration**: Seamless workflow for labeling on makesense.ai
- **Interactive Menu System**: User-friendly command-line interface
- **Automatic Configuration**: Generate YOLOv4-tiny config files based on your classes
- **Efficient Workflow**: Combined steps for faster processing
- **Complete Pipeline**: From video to training-ready dataset
- **Object Detection**: Run trained YOLOv4-tiny models on videos with real-time visualization

## Project Structure

```
Ai-Vision/
├── main.py                    # Main program with interactive menu
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore file
├── classes.txt               # Class names file (template/user data)
├── README.md                 # This file
│
├── app/                      # Application modules
│   ├── __init__.py          # Package initialization
│   ├── video_utils.py       # Video frame extraction utilities
│   ├── label_data.py        # Data labeling and YOLOv4-tiny preparation utilities
│   ├── objvision.py         # Object detection with YOLOv4-tiny
│   ├── video_input_handler.py # Centralized video input selection handler
│   ├── logger_config.py     # Logging configuration module
│   ├── exceptions.py        # Custom exception classes
│   └── constants.py         # Constants and configuration values
│
├── docs/                     # Documentation folder
│   ├── README.md            # Documentation index
│   ├── WORKFLOW.md          # Detailed workflow explanation
│   ├── OBJVISION_GUIDE.md   # Object detection guide
│   └── FILES_TO_REMOVE.md   # Cleanup documentation
│
├── yolov4-tiny/             # YOLOv4-tiny configuration
│   └── obj.names            # Class names for YOLO (generated)
│
└── media/                   # Media files directory (user data, gitignored)
    ├── images/              # Extracted frames and labels (for makesense.ai)
    ├── shuffled_images/     # Shuffled images (optional)
    ├── obj/                 # Training dataset folder
    ├── obj.zip              # Training dataset zip file
    └── *.mp4                # Video files
```

## Installation

### Recommended: Using Virtual Environment

Using a virtual environment is **highly recommended** to avoid conflicts with other Python projects.

#### Setup:

**Windows:**
```bash
# Create virtual environment
python -m venv vision

# Activate virtual environment
vision\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv vision

# Activate virtual environment
source vision/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Activating Virtual Environment (after setup):

**Windows:**
```bash
vision\Scripts\activate
```

**Linux/Mac:**
```bash
source vision/bin/activate
```

#### Deactivate (when done):
```bash
deactivate
```

### Alternative: Global Installation

If you prefer not to use a virtual environment:

```bash
pip install -r requirements.txt
```

**Note**: This will install packages globally, which may cause conflicts with other projects.

### Dependencies

- `opencv-python>=4.5.0` - Video playback and frame extraction
- `numpy>=1.19.0` - Numerical operations for image processing
- `Pillow>=8.0.0` - Image manipulation and saving
- `jupyter>=1.0.0` - Optional, for working with notebooks locally

**Note**: Python 3.9+ is recommended for full type hint support.

## Quick Start

### Running the Main Program

The easiest way to use this tool is through the interactive menu:

```bash
python main.py
```

This will guide you through the complete workflow step by step.

### Learning the Code

For a detailed explanation of how the code works, see the [Workflow Documentation](docs/WORKFLOW.md) in the `docs/` folder. It provides:
- Step-by-step code explanation
- Module details and data flow
- Key concepts and learning exercises
- Best practices

## Complete Workflow

### Step 1: Extract Frames & Prepare for Makesense.ai
Extract frames from various video sources with a configurable interval (default: every 30 frames), and automatically prepare them for makesense.ai. This step supports multiple video input options:

**Video Input Options:**
1. **Select from media folder** - Choose from videos in the `media/` folder
2. **Enter custom video file path** - Use any video file on your system
3. **Use webcam/camera** - Capture frames directly from your camera (with frame limit option)
4. **Video stream URL** - Extract from live streams (http/https/rtsp/rtmp/udp)

**Features:**
- Extracts frames from video directly to `media/images/` folder
- Configurable frame interval (extract every Nth frame)
- For webcam/stream: Option to set maximum frames to extract
- Supports all common video formats and streaming protocols

**Next**: 
1. Go to https://www.makesense.ai/
2. Upload images from `media/images/` folder
3. Label your objects
4. Export labels in YOLO format
5. Download and extract labels to `media/images/` folder (same folder as images)
   - **Important**: When exporting from makesense.ai, make sure to include the `classes.txt` file
   - The `classes.txt` file contains your class names and will be automatically detected

### Step 2: Process Labels & Prepare Training Data
This step automates the entire post-labeling workflow:
- **Process Labels**: Verifies all images have corresponding label files
- **Auto-Extract Classes**: Automatically extracts class names from makesense.ai export (from `classes.txt` file)
  - If classes are found automatically, you can confirm or edit them
  - If not found, you'll be prompted to enter class names
- **Update Configuration**: Automatically updates YOLOv4-tiny config files and creates `classes.txt`
- **Prepare Training Data**: Creates a zip file (`media/obj.zip`) with all labeled images and annotations

**Result**: You'll have a ready-to-use training dataset with:
- `media/obj.zip` file containing images and labels
- Updated YOLOv4 configuration files
- Class names saved for reference

**Next**:
1. Upload `media/obj.zip` to your training environment (Google Colab, local machine, etc.)
2. Use the zip file with your YOLOv4 training setup
3. Follow your training workflow
4. Download trained weights after training completes

### Step 3: Run Object Detection on Video (Optional)
After training your model in Google Colab and downloading the weights file:
1. Place `yolov4-tiny-custom_last.weights` in the project root directory
2. Run `python main.py` and select option 4
3. Choose video input method (same 4 options as Step 1):
   - Select from media folder
   - Enter custom video file path
   - Use webcam/camera
   - Video stream URL (http/https/rtsp/rtmp)
4. Configure detection settings (confidence threshold, save output)
5. View real-time detections with bounding boxes and labels

**Features**:
- Real-time object detection visualization
- Pause/Resume during playback (press 'p')
- Save output video with detections drawn
- Automatic path resolution (works from any directory)

## Advanced Usage

### Using Individual Modules

You can also use the modules individually:

### Video Frame Extraction

```python
from app.video_utils import VideoFrameExtractor

# Extract frames from video file
extractor = VideoFrameExtractor("media/Sheep.mp4", "images", frame_interval=30)
extractor.extract_frames()

# Extract frames from webcam (with frame limit)
extractor = VideoFrameExtractor(0, "images", frame_interval=30)  # 0 = default webcam
extractor.extract_frames(max_frames=100)  # Extract maximum 100 frames

# Extract frames from video stream
extractor = VideoFrameExtractor("rtsp://example.com/stream", "images", frame_interval=30)
extractor.extract_frames(max_frames=200)  # Extract maximum 200 frames

# Play video (optional)
extractor = VideoFrameExtractor("media/Sheep.mp4", "images")
extractor.play_video()
```

### Label Processing & Training Preparation

```python
from app.label_data import LabelUtils

lbUtils = LabelUtils()

# Process labels from makesense.ai
img_count, lbl_count = lbUtils.process_makesense_labels("media/img")

# Update classes and config
classes = ["sheep", "herder", "dog"]
lbUtils.update_config_files(classes)
lbUtils.save_classes_to_file(classes)

# Create training zip
zip_path = lbUtils.create_labeled_images_zip_file("media/img")
```

### Object Detection on Video

```python
from app.objvision import process_video

# Process a video file with trained model
detections = process_video(
    video_path="media/Sheep.mp4",
    cfg_file="./yolov4-tiny/yolov4-tiny-custom.cfg",
    weights_file="yolov4-tiny-custom_last.weights",
    confidence_threshold=0.5,
    save_output=True,
    output_path="media/Sheep_detected.mp4"
)

# Process webcam feed
detections = process_video(
    video_path=0,  # Camera index 0
    cfg_file="./yolov4-tiny/yolov4-tiny-custom.cfg",
    weights_file="yolov4-tiny-custom_last.weights",
    confidence_threshold=0.5
)

# Process video stream
detections = process_video(
    video_path="rtsp://example.com/stream",
    cfg_file="./yolov4-tiny/yolov4-tiny-custom.cfg",
    weights_file="yolov4-tiny-custom_last.weights",
    confidence_threshold=0.5
)
```

## Configuration

### Frame Extraction Settings

- **Frame Interval**: Extract every Nth frame (default: 30)
  - Lower values = more frames (slower, larger dataset)
  - Higher values = fewer frames (faster, smaller dataset)

### YOLOv4-tiny Settings

The `update_config_files()` method automatically calculates:
- **Number of filters**: `(number_of_classes + 5) * 3`
- **Max batches**: `max(6000, number_of_classes * 2000)`
- **Steps**: 80% and 90% of max_batches

These are automatically configured based on your number of classes.

## Requirements

- Python 3.6+
- Video files in `media/` folder for processing

## Tips & Best Practices

1. **Frame Extraction**: 
   - Start with interval 30-60 for most videos
   - Adjust based on video FPS and desired dataset size
   - More frames = better coverage but more labeling work

2. **Labeling on Makesense.ai**:
   - Be consistent with class names
   - Label all visible objects of interest
   - Use clear, descriptive class names

3. **Training**:
   - Monitor loss values - they should decrease over time
   - Training may take several hours for large datasets
   - Save checkpoints regularly to avoid losing progress

4. **Dataset Quality**:
   - Aim for at least 100-200 images per class
   - Ensure good variety in lighting, angles, backgrounds
   - Balance classes if possible (similar number of examples per class)

## Troubleshooting

**No video files found**: Place video files (.mp4, .avi, etc.) in the `media/` folder

**Labels not processing**: Ensure label files (.txt) are in YOLO format and match image filenames

**Config errors**: Make sure classes are set before generating config files

**Training data issues**: Check that `media/obj.zip` contains both images and label files, and that all paths are correct

## Author

Irfanhandrian

## License

This project is provided as-is for educational and research purposes.

