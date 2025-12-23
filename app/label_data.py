import os
import random
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Union
from app.logger_config import get_logger
from app.exceptions import LabelProcessingError, ConfigurationError
from app.constants import (
    DEFAULT_IMAGES_DIR,
    DEFAULT_OBJ_DIR,
    DEFAULT_CLASSES_FILE,
    DEFAULT_CONFIG_DIR,
    DEFAULT_MEDIA_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    CLASSES_TXT_FILENAME,
    calculate_filters,
    calculate_max_batches,
)

logger = get_logger(__name__)


class LabelUtils:

    def create_shuffled_images_folder(self, source_dir: Union[str, Path] = DEFAULT_IMAGES_DIR) -> None:
        """Shuffle images to prevent bias in training data."""
        source_path = Path(source_dir)
        shuffled_dir = DEFAULT_MEDIA_DIR / "shuffled_images"
        shuffled_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Shuffling images from {source_path} to {shuffled_dir}")
        
        image_files = [f for f in source_path.iterdir() if f.suffix.lower() == '.jpg']
        random.shuffle(image_files)
        
        for idx, img in enumerate(image_files):
            new_name = shuffled_dir / f"img_{idx:05d}.jpg"
            img.rename(new_name)
        
        logger.info(f"Shuffled {len(image_files)} images to {shuffled_dir}")

    def prepare_for_makesense(
        self,
        source_dir: Union[str, Path] = DEFAULT_IMAGES_DIR,
        output_dir: Optional[Union[str, Path]] = None
    ) -> int:
        """
        Prepare images for makesense.ai labeling.
        Note: This method is kept for compatibility but images are already in the correct location.
        """
        source_path = Path(source_dir)
        if output_dir is None:
            output_path = source_path
        else:
            output_path = Path(output_dir)
        
        if not source_path.exists():
            error_msg = f"Source directory not found: {source_dir}"
            logger.error(error_msg)
            raise LabelProcessingError(error_msg)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = [
            f for f in source_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        
        # Only copy if source and output are different
        if source_path != output_path:
            logger.info(f"Copying {len(image_files)} images to {output_path}...")
            print(f"Copying {len(image_files)} images to {output_dir}...")
            for img in image_files:
                shutil.copy2(img, output_path / img.name)
        else:
            logger.info(f"Found {len(image_files)} images in {output_path}")
            print(f"Found {len(image_files)} images in {output_dir}")
        
        logger.info(f"Images ready for makesense.ai! Upload the folder: {output_path}")
        print(f"Images ready for makesense.ai! Upload the folder: {output_dir}")
        return len(image_files)

    def extract_classes_from_labels(
        self,
        labels_dir: Union[str, Path] = DEFAULT_IMAGES_DIR
    ) -> Optional[List[str]]:
        """
        Extract class names from makesense.ai export.
        Makesense.ai typically exports a classes.txt file with the labels.
        
        Args:
            labels_dir: Directory containing label files from makesense.ai
        
        Returns:
            List of class names, or None if not found
        """
        labels_path = Path(labels_dir)
        
        # Check for classes.txt in labels directory (makesense.ai export)
        classes_file = labels_path / CLASSES_TXT_FILENAME
        if classes_file.exists():
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f if line.strip()]
                if classes:
                    logger.info(f"Found classes.txt in labels directory with {len(classes)} classes")
                    print(f"✓ Found classes.txt in labels directory with {len(classes)} classes")
                    return classes
            except Exception as e:
                logger.warning(f"Could not read classes.txt: {e}")
                print(f"Warning: Could not read classes.txt: {e}")
        
        # Check for classes.txt in project root
        project_root_classes = DEFAULT_CLASSES_FILE
        if project_root_classes.exists():
            try:
                with open(project_root_classes, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f if line.strip()]
                if classes:
                    logger.info(f"Found existing classes.txt in project root with {len(classes)} classes")
                    print(f"✓ Found existing classes.txt in project root with {len(classes)} classes")
                    return classes
            except Exception as e:
                logger.warning(f"Could not read classes.txt: {e}")
                print(f"Warning: Could not read classes.txt: {e}")
        
        # Try to infer number of classes from label files
        label_files = [
            f for f in labels_path.iterdir()
            if f.is_file() and f.suffix == '.txt' and f.name != CLASSES_TXT_FILENAME
        ]
        if label_files:
            max_class_id = -1
            for lbl_file in label_files[:10]:  # Check first 10 files to infer
                try:
                    with open(lbl_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                max_class_id = max(max_class_id, class_id)
                except Exception:
                    continue
            
            if max_class_id >= 0:
                num_classes = max_class_id + 1
                logger.info(f"Inferred {num_classes} classes from label files (class IDs 0-{max_class_id})")
                print(f"✓ Inferred {num_classes} classes from label files (class IDs 0-{max_class_id})")
                # Return None to let user provide names
                return None
        
        logger.debug("Could not extract classes from labels")
        return None
    
    def process_makesense_labels(
        self,
        labels_dir: Union[str, Path] = DEFAULT_IMAGES_DIR,
        classes: Optional[List[str]] = None
    ) -> Tuple[int, int, Optional[List[str]]]:
        """
        Process labels downloaded from makesense.ai.
        Makesense.ai exports YOLO format annotations as .txt files.
        
        Args:
            labels_dir: Directory containing images and label files from makesense.ai
            classes: List of class names in order (optional, will try to infer from labels)
        
        Returns:
            Tuple of (image_count, label_count, extracted_classes)
        """
        labels_path = Path(labels_dir)
        if not labels_path.exists():
            error_msg = f"Labels directory not found: {labels_dir}"
            logger.error(error_msg)
            raise LabelProcessingError(error_msg)
        
        # Find all image files
        image_files = [
            f for f in labels_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        
        # Find corresponding label files (exclude classes.txt)
        label_files = [
            f for f in labels_path.iterdir()
            if f.is_file() and f.suffix == '.txt' and f.name != CLASSES_TXT_FILENAME
        ]
        
        logger.info(f"Found {len(image_files)} images and {len(label_files)} label files")
        print(f"Found {len(image_files)} images and {len(label_files)} label files")
        
        # Verify all images have corresponding labels
        missing_labels = []
        for img in image_files:
            img_base = img.stem
            if not any(lbl.stem == img_base for lbl in label_files):
                missing_labels.append(img.name)
        
        if missing_labels:
            logger.warning(f"{len(missing_labels)} images without labels will be skipped")
            print(f"Warning: {len(missing_labels)} images without labels will be skipped")
        
        # Try to extract classes automatically
        extracted_classes = None
        if classes is None:
            extracted_classes = self.extract_classes_from_labels(labels_dir)
        
        return len(image_files), len(label_files), extracted_classes

    def create_labeled_images_zip_file(
        self,
        source_dir: Union[str, Path] = DEFAULT_IMAGES_DIR,
        output_dir: Union[str, Path] = DEFAULT_OBJ_DIR
    ) -> Path:
        """
        Create zip file for YOLOv4 training from labeled images.
        Handles both media/shuffled_images and media/images directories.
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            # Try shuffled_images as fallback
            shuffled_dir = DEFAULT_MEDIA_DIR / "shuffled_images"
            if shuffled_dir.exists():
                source_path = shuffled_dir
                logger.info(f"Using {source_path} as source directory")
                print(f"Using {source_path} as source directory")
            else:
                error_msg = f"Source directory not found: {source_dir}"
                logger.error(error_msg)
                raise LabelProcessingError(error_msg)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = [
            f for f in source_path.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        
        # Copy images and their corresponding label files
        copied_count = 0
        for img in image_files:
            img_base = img.stem
            txt_path = source_path / f"{img_base}.txt"
            
            # Copy image
            shutil.copy2(img, output_path / img.name)
            
            # Copy label if exists
            if txt_path.exists():
                shutil.copy2(txt_path, output_path / f"{img_base}.txt")
                copied_count += 1
            else:
                logger.warning(f"No label file found for {img.name}")
                print(f"Warning: No label file found for {img.name}")
        
        logger.info(f"Copied {copied_count} image-label pairs to {output_path}")
        print(f"Copied {copied_count} image-label pairs to {output_dir}")
        
        # Create zip file in media folder
        zip_path = DEFAULT_MEDIA_DIR / "obj.zip"
        if zip_path.exists():
            zip_path.unlink()
        
        logger.info(f"Creating zip file: {zip_path}")
        print(f"Creating zip file: {zip_path}")
        # Create zip from media/obj directory, archive will be media/obj.zip
        shutil.make_archive(str(DEFAULT_MEDIA_DIR / "obj"), 'zip', DEFAULT_MEDIA_DIR, 'obj')
        logger.info(f"Zip file created: {zip_path}")
        print(f"Zip file created: {zip_path}")
        
        return zip_path

    def update_config_files(
        self,
        classes: List[str],
        config_dir: Union[str, Path] = DEFAULT_CONFIG_DIR
    ) -> Path:
        """
        Update YOLOv4-tiny configuration files with class information.
        
        Args:
            classes: List of class names
            config_dir: Directory containing YOLOv4-tiny config files
        
        Returns:
            Path to the created names file
        """
        if not classes:
            error_msg = "Classes list cannot be empty"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        config_path = Path(config_dir)
        config_path.mkdir(parents=True, exist_ok=True)
        
        # Create obj.names file
        names_file = config_path / "obj.names"
        with open(names_file, "w", encoding='utf-8') as file:
            file.write("\n".join(classes))
        logger.info(f"Created {names_file} with {len(classes)} classes")
        print(f"Created {names_file} with {len(classes)} classes")
        
        # Update config file if template exists
        template_file = config_path / "yolov4-tiny-custom_template.cfg"
        if template_file.exists():
            with open(template_file, 'r', encoding='utf-8') as file:
                cfg_content = file.read()
            
            num_classes = len(classes)
            num_filters = calculate_filters(num_classes)
            max_batches = calculate_max_batches(num_classes)
            
            updated_cfg_content = cfg_content.replace('_CLASS_NUMBER_', str(num_classes))
            updated_cfg_content = updated_cfg_content.replace('_NUMBER_OF_FILTERS_', str(num_filters))
            updated_cfg_content = updated_cfg_content.replace('_MAX_BATCHES_', str(max_batches))
            
            output_file = config_path / "yolov4-tiny-custom.cfg"
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(updated_cfg_content)
            
            logger.info(f"Created {output_file} with {num_classes} classes, {num_filters} filters, {max_batches} batches")
            print(f"Created {output_file}")
            print(f"  - Classes: {num_classes}")
            print(f"  - Filters: {num_filters}")
            print(f"  - Max batches: {max_batches}")
        else:
            logger.warning(f"Template file not found: {template_file}")
            print(f"Warning: Template file not found: {template_file}")
            print("You may need to create the config file manually.")
        
        return names_file

    def save_classes_to_file(
        self,
        classes: List[str],
        filename: Union[str, Path] = DEFAULT_CLASSES_FILE
    ) -> Path:
        """Save class names to a text file for easy reference."""
        file_path = Path(filename)
        with open(file_path, "w", encoding='utf-8') as file:
            file.write("\n".join(classes))
        logger.info(f"Classes saved to {file_path}")
        print(f"Classes saved to {filename}")
        return file_path

# If you're not going to label all the generated images, make sure to shuffle them.
# Shuffling helps ensure that you will cover a wide range of scenarios. 
# This avoids any bias towards specific patterns or sequences.

# The function below shuffles the images in the images folder and inserts them into the shuffled_images folder.
"""
lbUtils = LabelUtils()
lbUtils.create_shuffled_images_folder()
"""
