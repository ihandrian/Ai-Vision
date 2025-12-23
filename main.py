"""
Object Detection Dataset Generator - Main Program
Orchestrates the complete workflow from video to YOLOv4 training preparation.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Union
from datetime import datetime

from app.video_utils import VideoFrameExtractor, find_video_files
from app.label_data import LabelUtils
from app.video_input_handler import VideoInputHandler
from app.logger_config import setup_logger, get_logger
from app.exceptions import (
    ObjectDetectionError,
    VideoProcessingError,
    LabelProcessingError,
    InvalidInputError,
)
from app.constants import (
    DEFAULT_MEDIA_DIR,
    DEFAULT_IMAGES_DIR,
    DEFAULT_FRAME_INTERVAL,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CLASSES_FILE,
    DEFAULT_CFG_FILE,
    DEFAULT_WEIGHTS_FILE,
    MENU_EXTRACT_FRAMES,
    MENU_PROCESS_LABELS,
    MENU_VIEW_CLASSES,
    MENU_RUN_DETECTION,
    MENU_EXIT,
    MAKESENSE_URL,
    OUTPUT_VIDEO_SUFFIX,
    CAMERA_OUTPUT_PREFIX,
    STREAM_OUTPUT_PREFIX,
)

# Setup logger
logger = setup_logger("obj_detection", log_level=logging.INFO)

# Optional import for detection feature
try:
    from app.objvision import process_video
    DETECTION_AVAILABLE = True
    logger.debug("Detection module imported successfully")
except ImportError as e:
    DETECTION_AVAILABLE = False
    DETECTION_IMPORT_ERROR = str(e)
    logger.warning(f"Detection module not available: {e}")


class ObjDetection:
    """Main workflow orchestrator for object detection dataset creation."""
    
    def __init__(
        self,
        media_dir: Union[str, Path] = DEFAULT_MEDIA_DIR,
        images_dir: Union[str, Path] = DEFAULT_IMAGES_DIR
    ):
        """
        Initialize the object detection workflow.
        
        Args:
            media_dir: Directory containing media files
            images_dir: Directory for extracted images
        """
        self.label_utils = LabelUtils()
        self.classes: List[str] = []
        self.media_dir = Path(media_dir)
        self.images_dir = Path(images_dir)
        self.video_input_handler = VideoInputHandler(self.media_dir)
        logger.info(f"Initialized ObjDetection: media_dir={self.media_dir}, images_dir={self.images_dir}")
        
    def display_menu(self) -> None:
        """Display main menu options."""
        print("\n" + "="*60)
        print("  Object Detection Dataset Generator")
        print("="*60)
        print(f"{MENU_EXTRACT_FRAMES}. Extract frames from video & prepare for makesense.ai")
        print(f"{MENU_PROCESS_LABELS}. Process labels & prepare training data (auto-extracts classes)")
        print(f"{MENU_VIEW_CLASSES}. View current classes")
        print(f"{MENU_RUN_DETECTION}. Run object detection on video (requires trained model)")
        print(f"{MENU_EXIT}. Exit")
        print("="*60)
    
    def extract_and_prepare_frames(self) -> None:
        """Step 1: Extract frames from video and prepare for makesense.ai."""
        logger.info("Starting frame extraction workflow")
        print("\n--- Step 1: Extract Frames & Prepare for makesense.ai ---")
        
        try:
            # Get video input from user using VideoInputHandler
            self.video_input_handler.display_video_input_options()
            selected_video, input_type = self.video_input_handler.get_video_input(require_confirmation=True)
        except (InvalidInputError, VideoProcessingError) as e:
            logger.error(f"Video input error: {e}")
            print(f"\nError: {e}")
            return
        
        try:
            # Ask for frame interval
            interval_input = input(f"\nFrame interval (extract every Nth frame, default {DEFAULT_FRAME_INTERVAL}): ").strip()
            try:
                interval = int(interval_input) if interval_input else DEFAULT_FRAME_INTERVAL
                if interval <= 0:
                    raise ValueError("Frame interval must be positive")
            except ValueError as e:
                logger.warning(f"Invalid frame interval '{interval_input}', using default: {e}")
                print(f"Invalid input, using default {DEFAULT_FRAME_INTERVAL}")
                interval = DEFAULT_FRAME_INTERVAL
            
            # Ask for max frames if using webcam or stream
            max_frames: Optional[int] = None
            is_camera = self.video_input_handler.is_camera(selected_video)
            is_stream = self.video_input_handler.is_stream(selected_video)
            
            if is_camera or is_stream:
                max_input = input("Maximum frames to extract (press Enter for unlimited, Ctrl+C to stop): ").strip()
                if max_input:
                    try:
                        max_frames = int(max_input)
                        if max_frames <= 0:
                            logger.warning(f"Invalid max_frames '{max_input}', using unlimited")
                            print("Invalid number, using unlimited frames")
                            max_frames = None
                    except ValueError:
                        logger.warning(f"Invalid max_frames input '{max_input}', using unlimited")
                        print("Invalid input, using unlimited frames")
                        max_frames = None
            
            # Extract frames directly to media/images
            logger.info(f"Extracting frames: video={selected_video}, interval={interval}, max_frames={max_frames}")
            extractor = VideoFrameExtractor(selected_video, self.images_dir, interval)
            count = extractor.extract_frames(max_frames=max_frames)
            
            logger.info(f"Successfully extracted {count} frames to {self.images_dir}")
            print(f"✓ Successfully extracted {count} frames to '{self.images_dir}' folder")
            
            print("\n" + "="*60)
            print("Next steps:")
            print(f"  1. Go to {MAKESENSE_URL}")
            print(f"  2. Upload the images from '{self.images_dir}' folder")
            print("  3. Label your objects")
            print("  4. Export labels in YOLO format")
            print(f"  5. Download and extract to '{self.images_dir}' folder")
            print("="*60)
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
            print(f"Invalid input: {e}")
        except VideoProcessingError as e:
            logger.error(f"Video processing error: {e}")
            print(f"Error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during frame extraction: {e}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def process_labels_and_prepare_training(self) -> None:
        """Step 2: Process labels, update classes, and prepare training data."""
        logger.info("Starting label processing workflow")
        print("\n--- Step 2: Process Labels & Prepare Training Data ---")
        
        labels_dir = self.images_dir
        if not labels_dir.exists():
            error_msg = f"Images directory '{labels_dir}' not found! Please extract frames from video first (Step 1)."
            logger.error(error_msg)
            print(f"Images directory '{labels_dir}' not found!")
            print("Please extract frames from video first (Step 1).")
            return
        
        # Step 2a: Process labels and extract classes
        print("\n[1/3] Processing labels from makesense.ai...")
        try:
            img_count, lbl_count, extracted_classes = self.label_utils.process_makesense_labels(labels_dir)
            logger.info(f"Processed {img_count} images and {lbl_count} label files")
            print(f"✓ Processed {img_count} images and {lbl_count} label files")
            
            if img_count != lbl_count:
                logger.warning(f"Mismatch between images ({img_count}) and labels ({lbl_count})")
                print(f"⚠ Warning: Mismatch between images ({img_count}) and labels ({lbl_count})")
        except LabelProcessingError as e:
            logger.error(f"Label processing error: {e}")
            print(f"Error processing labels: {e}")
            return
        except Exception as e:
            logger.exception(f"Unexpected error processing labels: {e}")
            print(f"Error processing labels: {e}")
            return
        
        # Step 2b: Handle classes (auto-extract or prompt user)
        print("\n[2/3] Setting up classes...")
        
        if extracted_classes:
            # Classes found automatically
            classes = extracted_classes
            print(f"✓ Using classes from makesense.ai export: {', '.join(classes)}")
            
            # Ask user to confirm or edit
            print(f"\nFound {len(classes)} classes:")
            for i, cls in enumerate(classes, 1):
                print(f"  {i}. {cls}")
            
            confirm = input("\nUse these classes? (y/n, default y): ").strip().lower()
            if confirm == 'n':
                # User wants to edit
                print("\nEnter class names (one per line, press Enter twice when done):")
                print("You can edit existing names or enter new ones.")
                classes = []
                for i in range(len(extracted_classes)):
                    default = extracted_classes[i]
                    class_name = input(f"  Class {i+1} (default: {default}): ").strip()
                    if not class_name:
                        class_name = default
                    classes.append(class_name)
                
                # Allow adding more classes
                while True:
                    class_name = input(f"  Class {len(classes) + 1} (or press Enter to finish): ").strip()
                    if not class_name:
                        break
                    classes.append(class_name)
        else:
            # No classes found, need user input
            print("Could not automatically detect classes from makesense.ai export.")
            print("Please enter the class names you used in makesense.ai")
            print("(Enter one class per line, press Enter twice when done)")
            
            classes = []
            print("\nClass names:")
            while True:
                class_name = input(f"  Class {len(classes) + 1}: ").strip()
                if not class_name:
                    if classes:
                        break
                    else:
                        print("Please enter at least one class!")
                        continue
                classes.append(class_name)
        
        self.classes = classes
        logger.info(f"Final classes set: {self.classes}")
        print(f"\n✓ Final classes: {', '.join(self.classes)}")
        
        # Save to file
        try:
            self.label_utils.save_classes_to_file(self.classes)
        except Exception as e:
            logger.error(f"Error saving classes to file: {e}")
            print(f"Warning: Could not save classes to file: {e}")
        
        # Update config files
        try:
            self.label_utils.update_config_files(self.classes)
            logger.info("Configuration files updated successfully")
            print("✓ Configuration files updated")
        except Exception as e:
            logger.warning(f"Could not update config files: {e}")
            print(f"Warning: Could not update config files: {e}")
        
        # Step 2c: Prepare training data
        print("\n[3/3] Preparing training data...")
        try:
            zip_path = self.label_utils.create_labeled_images_zip_file(labels_dir)
            logger.info(f"Training data prepared: {zip_path}")
            print(f"✓ Training data prepared: {zip_path}")
            print(f"✓ Classes: {', '.join(self.classes)}")
            
            print("\n" + "="*60)
            print("✓ All steps completed successfully!")
            print("="*60)
            print("\nYour training data is ready:")
            print(f"  - Training zip: {zip_path}")
            print(f"  - Classes: {len(self.classes)} ({', '.join(self.classes)})")
            print(f"  - Classes file: {DEFAULT_CLASSES_FILE}")
            print("\nNext: Upload the zip file to your training environment (Google Colab).")
            print("="*60)
        except LabelProcessingError as e:
            logger.error(f"Error preparing training data: {e}")
            print(f"Error preparing training data: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error preparing training data: {e}")
            print(f"Error preparing training data: {e}")
    
    def view_classes(self) -> None:
        """View current classes."""
        logger.info("Viewing current classes")
        print("\n--- Current Classes ---")
        if self.classes:
            print(f"Number of classes: {len(self.classes)}")
            for i, cls in enumerate(self.classes, 1):
                print(f"  {i}. {cls}")
        else:
            logger.info("No classes set yet")
            print("No classes set yet. Use Step 2 to process labels and set classes.")
    
    def run_detection(self) -> None:
        """Run object detection on video using trained YOLOv4-tiny model."""
        logger.info("Starting object detection workflow")
        print("\n--- Run Object Detection on Video ---")
        
        if not DETECTION_AVAILABLE:
            error_msg = f"Detection feature is not available. Import error: {DETECTION_IMPORT_ERROR}"
            logger.error(error_msg)
            print(f"Error: Detection feature is not available.")
            print(f"Import error: {DETECTION_IMPORT_ERROR}")
            print("Please ensure all dependencies are installed: pip install -r requirements.txt")
            return
        
        # Check if required files exist
        cfg_file = DEFAULT_CFG_FILE
        weights_file = DEFAULT_WEIGHTS_FILE
        
        if not cfg_file.exists():
            error_msg = f"Config file not found: {cfg_file}"
            logger.error(error_msg)
            print(f"Error: Config file not found: {cfg_file}")
            print("Please ensure you have completed training and have the config file.")
            return
        
        if not weights_file.exists():
            error_msg = f"Weights file not found: {weights_file}"
            logger.error(error_msg)
            print(f"Error: Weights file not found: {weights_file}")
            print("Please download the trained weights from Google Colab and place in project root.")
            return
        
        try:
            # Get video input from user using VideoInputHandler
            self.video_input_handler.display_video_input_options()
            video_path, input_type = self.video_input_handler.get_video_input(require_confirmation=False)
        except (InvalidInputError, VideoProcessingError) as e:
            logger.error(f"Video input error: {e}")
            print(f"\nError: {e}")
            return
        
        # Detection settings
        print("\nDetection Settings:")
        try:
            conf_input = input(f"Confidence threshold (0.0-1.0, default {DEFAULT_CONFIDENCE_THRESHOLD}): ").strip()
            confidence_threshold = float(conf_input) if conf_input else DEFAULT_CONFIDENCE_THRESHOLD
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        except ValueError as e:
            logger.warning(f"Invalid confidence threshold '{conf_input}', using default: {e}")
            print(f"Invalid confidence threshold, using default {DEFAULT_CONFIDENCE_THRESHOLD}")
            confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        
        save_output_input = input("Save output video? (y/n, default n): ").strip().lower()
        save_output = save_output_input == 'y'
        
        output_path: Optional[Path] = None
        if save_output:
            if isinstance(video_path, int):
                base_name = CAMERA_OUTPUT_PREFIX.format(video_path)
            elif self.video_input_handler.is_stream(video_path):
                # For streams, use a timestamp-based name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = STREAM_OUTPUT_PREFIX.format(timestamp)
            else:
                base_name = Path(video_path).stem
            output_path = self.media_dir / f"{base_name}{OUTPUT_VIDEO_SUFFIX}"
            logger.info(f"Output will be saved to: {output_path}")
            print(f"Output will be saved to: {output_path}")
        
        # Process video
        print("\n" + "="*60)
        if isinstance(video_path, int):
            logger.info(f"Starting detection from camera (index {video_path})...")
            print(f"Starting detection from camera (index {video_path})...")
        elif self.video_input_handler.is_stream(video_path):
            logger.info(f"Starting detection from video stream: {video_path}")
            print(f"Starting detection from video stream...")
            print(f"Stream URL: {video_path}")
        else:
            video_name = Path(video_path).name
            logger.info(f"Processing video: {video_name}")
            print(f"Processing {video_name}...")
        print("Press 'q' during playback to quit, 'p' to pause")
        print("="*60)
        
        try:
            # Run detection
            logger.info(f"Running detection: confidence={confidence_threshold}, save_output={save_output}")
            detections = process_video(
                video_path=video_path,
                cfg_file=str(cfg_file),
                weights_file=str(weights_file),
                names_file=None,  # Auto-detect
                output_path=str(output_path) if output_path else None,
                confidence_threshold=confidence_threshold,
                display=True,
                save_output=save_output
            )
            
            # Show summary
            total_detections = sum(len(d['detections']) for d in detections)
            logger.info(f"Detection complete: {len(detections)} frames, {total_detections} objects detected")
            print(f"\n{'='*60}")
            print(f"Detection Summary:")
            print(f"  - Total frames processed: {len(detections)}")
            print(f"  - Total objects detected: {total_detections}")
            if save_output and output_path:
                print(f"  - Output saved to: {output_path}")
            print(f"{'='*60}")
        except VideoProcessingError as e:
            logger.error(f"Video processing error during detection: {e}")
            print(f"Error during detection: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during detection: {e}")
            print(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
    
    def load_classes_from_file(self) -> None:
        """Try to load classes from file if it exists."""
        classes_file = DEFAULT_CLASSES_FILE
        if classes_file.exists():
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    self.classes = [line.strip() for line in f if line.strip()]
                if self.classes:
                    logger.info(f"Loaded {len(self.classes)} classes from {classes_file}")
                    print(f"Loaded {len(self.classes)} classes from {classes_file}")
            except Exception as e:
                logger.warning(f"Could not load classes from {classes_file}: {e}")
                print(f"Could not load classes: {e}")
    
    def run(self) -> None:
        """Main program loop."""
        logger.info("Starting object detection application")
        # Try to load existing classes
        self.load_classes_from_file()
        
        while True:
            self.display_menu()
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == MENU_EXTRACT_FRAMES:
                self.extract_and_prepare_frames()
            elif choice == MENU_PROCESS_LABELS:
                self.process_labels_and_prepare_training()
            elif choice == MENU_VIEW_CLASSES:
                self.view_classes()
            elif choice == MENU_RUN_DETECTION:
                self.run_detection()
            elif choice == MENU_EXIT:
                logger.info("User requested exit")
                print("\nExiting... Goodbye!")
                break
            else:
                logger.warning(f"Invalid menu option selected: {choice}")
                print("\nInvalid option! Please select 1-5.")
            
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        workflow = ObjDetection()
        workflow.run()
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        print("\n\nProgram interrupted by user. Goodbye!")
        sys.exit(0)
    except ObjectDetectionError as e:
        logger.error(f"Object detection error: {e}")
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

