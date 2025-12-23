# Object Detection Dataset Generator - Code Workflow Documentation

This document explains the code workflow in a systematic order, designed for learning purposes.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Entry Point: main.py](#entry-point-mainpy)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [Module Details](#module-details)
6. [Data Flow](#data-flow)
7. [Key Concepts](#key-concepts)

---

## Project Overview

This project automates the process of creating training datasets for YOLOv4 object detection models. It takes videos as input and produces a ready-to-use training dataset.

### High-Level Process

```
Video → Extract Frames → Label on makesense.ai → Process Labels → Training Dataset
```

---

## Architecture Overview

### Project Structure

```
ObjectDetection/
├── main.py                 # Entry point, orchestrates the workflow
├── app/
│   ├── __init__.py        # Package initialization
│   ├── video_utils.py     # Video processing utilities
│   └── label_data.py      # Label processing and dataset preparation
└── docs/
    └── WORKFLOW.md        # This file
```

### Module Responsibilities

- **main.py**: User interface and workflow orchestration
- **video_utils.py**: Video frame extraction
- **label_data.py**: Label processing, class management, dataset preparation

---

## Entry Point: main.py

### Program Initialization

When you run `python main.py`, the following happens:

1. **Import Dependencies**
   ```python
   from app.video_utils import VideoFrameExtractor, find_video_files
   from app.label_data import LabelUtils
   ```
   - Imports video processing utilities
   - Imports label processing utilities

2. **Create Workflow Object**
   ```python
   workflow = ObjDetection()
   ```
   - Initializes the main workflow class
   - Sets up default directories (`media/`, `images/`)
   - Creates `LabelUtils` instance

3. **Load Existing Classes** (if available)
   - Checks for `classes.txt` file
   - Loads previously saved class names

4. **Start Main Loop**
   - Displays menu
   - Waits for user input
   - Executes selected option
   - Repeats until user exits

---

## Step-by-Step Workflow

### Step 1: Extract Frames & Prepare for Makesense.ai

**Method**: `extract_and_prepare_frames()`

#### Process Flow:

1. **Find Video Files**
   ```python
   video_files = find_video_files(self.media_dir)
   ```
   - Scans `media/` folder for video files (.mp4, .avi, etc.)
   - Returns list of video file paths

2. **User Selects Video Input Method**
   The system now supports 4 video input options:
   
   **Option 1: Select from media folder**
   - Displays available videos in `media/` folder
   - User chooses which video to process
   - Defaults to first video if no selection
   
   **Option 2: Enter custom video file path**
   - User provides path to any video file on system
   - Supports absolute and relative paths
   - Validates file existence before processing
   
   **Option 3: Use webcam/camera**
   - Captures frames directly from camera
   - User can specify camera index (0 for default)
   - Option to set maximum frames to extract
   - Useful for live labeling scenarios
   
   **Option 4: Video stream URL**
   - Supports multiple streaming protocols:
     - HTTP/HTTPS: `http://example.com/video.mp4`
     - RTSP: `rtsp://username:password@ip:port/stream`
     - RTMP: `rtmp://server/live/stream`
     - UDP: `udp://@:port`
   - Option to set maximum frames to extract
   - Validates URL format before connecting

3. **Get Frame Interval**
   - Asks user for frame extraction interval (default: 30)
   - Lower number = more frames (slower, larger dataset)
   - Higher number = fewer frames (faster, smaller dataset)

4. **Get Maximum Frames (for webcam/stream only)**
   - Optional limit for webcam or stream sources
   - Prevents unlimited frame extraction
   - User can press Ctrl+C to stop extraction manually

5. **Extract Frames**
   ```python
   extractor = VideoFrameExtractor(selected_video, self.images_dir, interval)
   count = extractor.extract_frames(max_frames=max_frames)
   ```
   - Creates `VideoFrameExtractor` instance
   - Handles different video source types (file, camera, stream)
   - Extracts frames at specified interval
   - Saves frames directly to `media/images/` folder as `img_00000.jpg`, `img_00001.jpg`, etc.

6. **Images Ready for Makesense.ai**
   - Images are already in `media/images/` folder
   - No copying needed - ready for upload to makesense.ai

#### What Happens Behind the Scenes:

**VideoFrameExtractor.extract_frames()**:
- Detects video source type (file, camera, or stream)
- Opens video source using OpenCV (supports files, cameras, and URLs)
- Reads frames one by one
- Saves every Nth frame (based on interval)
- For webcam/stream: Respects max_frames limit if set
- Handles KeyboardInterrupt for manual stopping
- Tracks progress (prints every 10 frames)
- Returns total count of extracted frames

**Images are stored directly in `media/images/`**:
- Frames are extracted directly to `media/images/`
- No copying step needed
- Same folder will contain labels after downloading from makesense.ai

---

### Step 2: Process Labels, Update Classes & Prepare Training Data

**Method**: `process_labels_and_prepare_training()`

This is a **combined step** that does three things in sequence:

#### Sub-step 2a: Process Labels

**Process Flow:**

1. **Check Labels Directory**
   ```python
   labels_dir = "media/images"
   if not os.path.exists(labels_dir):
       # Error: labels not found
   ```
   - Verifies that `media/images/` exists
   - Should contain images AND label files (.txt) from makesense.ai

2. **Process Labels**
   ```python
   img_count, lbl_count = self.label_utils.process_makesense_labels(labels_dir)
   ```
   - Scans directory for image files
   - Scans directory for label files (.txt)
   - Verifies each image has a corresponding label file
   - Returns counts of images and labels

3. **Validate**
   - Checks if image count matches label count
   - Warns if there's a mismatch

#### Sub-step 2b: Update Classes

**Process Flow:**

1. **Prompt User for Classes**
   ```python
   classes = []
   while True:
       class_name = input(f"  Class {len(classes) + 1}: ").strip()
       if not class_name:
           break
       classes.append(class_name)
   ```
   - Asks user to enter class names one by one
   - Continues until user presses Enter twice
   - Example: ["sheep", "herder", "dog"]

2. **Save Classes**
   ```python
   self.classes = classes
   self.label_utils.save_classes_to_file(self.classes)
   ```
   - Stores classes in workflow object
   - Saves to `classes.txt` file for future reference

3. **Update Configuration Files**
   ```python
   self.label_utils.update_config_files(self.classes)
   ```
   - Creates `yolov4-tiny/obj.names` file with class names
   - Calculates configuration values:
     - Number of classes
     - Number of filters: `(classes + 5) * 3`
     - Max batches: `max(6000, classes * 2000)`
   - Updates YOLOv4 config file (if template exists)

#### Sub-step 2c: Prepare Training Data

**Process Flow:**

1. **Create Training Dataset**
   ```python
   zip_path = self.label_utils.create_labeled_images_zip_file(labels_dir)
   ```
   - Scans `media/images/` for image-label pairs
   - Copies matching pairs to `media/obj/` directory
   - Creates `media/obj.zip` file containing all training data

2. **Verify Results**
   - Confirms zip file creation
   - Displays class information
   - Shows summary of what was created

#### What Happens Behind the Scenes:

**LabelUtils.process_makesense_labels()**:
- Lists all files in directory
- Separates images from labels
- Checks for matching pairs (same filename, different extension)
- Returns counts and warnings

**LabelUtils.update_config_files()**:
- Creates `obj.names` file (one class per line)
- Calculates YOLOv4 parameters based on class count
- Updates config file with calculated values

**LabelUtils.create_labeled_images_zip_file()**:
- Finds all image files
- For each image, looks for corresponding .txt label file
- Copies both to `media/obj/` directory
- Creates zip archive of `media/obj/` directory
- Returns path to zip file (`media/obj.zip`)

---

### Step 3: View Current Classes

**Method**: `view_classes()`

Simple utility to display currently loaded classes:
- Shows number of classes
- Lists all class names with numbers
- Useful for verification

---

## Module Details

### app/video_utils.py

#### VideoFrameExtractor Class

**Purpose**: Extract frames from video files

**Key Methods**:

1. **`__init__(video_path, output_dir, frame_interval)`**
   - Stores video path, output directory, and interval
   - Creates output directory if it doesn't exist

2. **`extract_frames()`**
   - Opens video using `cv2.VideoCapture()`
   - Reads frames sequentially
   - Saves frames at specified interval
   - Returns count of saved frames

3. **`play_video()`** (optional)
   - Plays video in a window
   - Useful for preview
   - Press ESC to exit

**Helper Function**:

- **`find_video_files(media_dir)`**
  - Scans directory for video files
  - Returns list of video file paths
  - Supports: .mp4, .avi, .mov, .mkv, .flv, .wmv

---

### app/label_data.py

#### LabelUtils Class

**Purpose**: Process labels and prepare training datasets

**Key Methods**:

1. **`prepare_for_makesense(source_dir, output_dir)`**
   - Copies images to makesense.ai-ready directory
   - Creates clean folder structure

2. **`process_makesense_labels(labels_dir)`**
   - Validates label files
   - Checks for image-label pairs
   - Returns counts and warnings

3. **`update_config_files(classes)`**
   - Creates `obj.names` file
   - Calculates YOLOv4 parameters
   - Updates configuration files

4. **`create_labeled_images_zip_file(source_dir, output_dir)`**
   - Copies image-label pairs to output directory
   - Creates zip file for training
   - Returns path to zip file

5. **`save_classes_to_file(classes, filename)`**
   - Saves class names to text file
   - Useful for persistence

---

## Data Flow

### Complete Data Flow Diagram

```
┌─────────────┐
│   Video     │
│  (media/)   │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Extract Frames      │
│  (video_utils.py)    │
└──────┬──────────────┘
       │
       ▼
┌─────────────┐
│   Images    │
│(media/images/)│
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Upload to          │
│  makesense.ai       │
│  (Manual Step)      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Download Labels    │
│  to media/images/   │
│  (Manual Step)      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Process Labels     │
│  Update Classes     │
│  Create media/obj.zip│
│  (label_data.py)     │
└──────┬──────────────┘
       │
       ▼
┌─────────────┐
│ media/obj.zip│
│  (Ready!)   │
└─────────────┘
```

### File Structure Evolution

**Initial State:**
```
media/
  └── Sheep.mp4
```

**After Step 1:**
```
media/
  ├── images/
  │   ├── img_00000.jpg
  │   ├── img_00001.jpg
  │   └── ...
  └── Sheep.mp4
```

**After Labeling (Manual):**
```
media/
  └── images/
      ├── img_00000.jpg
      ├── img_00000.txt  (from makesense.ai)
      ├── img_00001.jpg
      ├── img_00001.txt  (from makesense.ai)
      └── ...
```

**After Step 2:**
```
media/
  └── obj/
      ├── img_00000.jpg
      ├── img_00000.txt
      ├── img_00001.jpg
      ├── img_00001.txt
      └── ...
media/obj.zip  (created)
classes.txt  (created)
yolov4-tiny/
  └── obj.names  (created)
```

---

## Key Concepts

### 1. Video Input Options

**Supported Sources:**
- **Media Folder**: Local video files in `media/` directory
- **Custom Path**: Any video file on your system (absolute or relative path)
- **Webcam/Camera**: Direct camera capture (useful for live labeling)
- **Video Stream**: Live streams via HTTP/HTTPS/RTSP/RTMP/UDP protocols

**When to use each:**
- **Media Folder**: Standard workflow with local video files
- **Custom Path**: When videos are stored elsewhere on your system
- **Webcam**: For real-time frame capture and labeling
- **Stream**: For processing live video feeds or remote video sources

**Important Notes:**
- Webcam and stream sources should use frame limits to prevent excessive extraction
- Stream extraction depends on network stability and stream availability
- Camera index 0 is typically the default webcam

### 2. Frame Extraction Interval

**What it means:**
- Extract every Nth frame from the video
- Example: interval=30 means extract frames 0, 30, 60, 90, etc.

**Why it matters:**
- Too low (e.g., 5): Many similar frames, large dataset, slow processing
- Too high (e.g., 100): Few frames, may miss important moments
- Recommended: 30-60 for most videos

### 3. YOLO Label Format

**Format:**
```
class_id center_x center_y width height
```

**Example:**
```
0 0.5 0.5 0.3 0.4
```
- All values are normalized (0-1)
- `center_x, center_y`: Center of bounding box
- `width, height`: Size of bounding box

**Why normalized:**
- Works with any image size
- Makes training more flexible

### 4. YOLOv4 Configuration

**Key Parameters:**

- **Classes**: Number of object classes
- **Filters**: Calculated as `(classes + 5) * 3`
  - Each detection needs: x, y, w, h, confidence + class probabilities
  - 3 anchor boxes per grid cell
- **Max Batches**: Training iterations
  - Formula: `max(6000, classes * 2000)`
  - More classes = more training needed

### 5. Dataset Structure

**Training Dataset Contains:**
- Images: `.jpg` files
- Labels: `.txt` files (YOLO format)
- One-to-one correspondence: `img_00000.jpg` ↔ `img_00000.txt`

**Why This Structure:**
- YOLOv4 expects this format
- Easy to process programmatically
- Standard in object detection workflows

---

## Error Handling

### Common Errors and Solutions

1. **"No video files found"**
   - **Cause**: No videos in `media/` folder
   - **Solution**: Add video files (.mp4, .avi, etc.) to `media/`

2. **"Labels directory not found"**
   - **Cause**: `media/img/` doesn't exist or is empty
   - **Solution**: Complete Step 1 first, then download labels from makesense.ai

3. **"Mismatch between images and labels"**
   - **Cause**: Some images don't have corresponding label files
   - **Solution**: Ensure all images are labeled and exported from makesense.ai

4. **"Classes not set"**
   - **Cause**: Classes weren't entered
   - **Solution**: Enter at least one class name when prompted

---

## Learning Exercises

### Exercise 1: Trace the Flow
1. Run the program with a test video
2. Follow each step and note what files are created
3. Check each directory after each step
4. Understand the data transformation at each stage

### Exercise 2: Modify Frame Interval
1. Extract frames with interval=10
2. Extract frames with interval=60
3. Compare the number of frames extracted
4. Understand the trade-off between dataset size and processing time

### Exercise 3: Understand Label Format
1. Open a `.txt` label file from makesense.ai
2. Understand what each number means
3. Try to visualize the bounding box on the image
4. Modify values and see how it affects detection

### Exercise 4: Explore Configuration
1. Check `classes.txt` after setting classes
2. Look at `yolov4-tiny/obj.names` file
3. Understand how class count affects YOLOv4 parameters
4. Research YOLOv4 configuration file structure

---

## Best Practices

1. **Start Small**: Test with a short video first
2. **Consistent Naming**: Use clear, descriptive class names
3. **Validate Data**: Always check image-label pairs match
4. **Backup**: Keep original videos and labels safe
5. **Document**: Note your frame intervals and class names

---

## Next Steps

After understanding this workflow:

1. **Learn YOLOv4 Training**: Understand how to use `media/obj.zip` for training
2. **Explore makesense.ai**: Learn efficient labeling techniques
3. **Optimize Parameters**: Experiment with frame intervals and class counts
4. **Extend Functionality**: Add features like data augmentation, validation splits

---

## Additional Resources

- **YOLOv4 Paper**: Understanding the model architecture
- **OpenCV Documentation**: Video processing functions
- **makesense.ai Guide**: Efficient labeling workflows
- **Object Detection Basics**: Understanding bounding boxes and annotations

---

*This documentation is designed for learning purposes. For specific implementation details, refer to the source code comments.*

