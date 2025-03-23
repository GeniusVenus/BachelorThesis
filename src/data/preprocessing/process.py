import re
import cv2
import os
import numpy as np
from patchify import patchify
import splitfolders
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class DatasetProcessor:
    """A class to handle dataset processing operations for image segmentation tasks with optimized performance."""

    def __init__(self, config):
        """
        Initialize the DatasetProcessor with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary containing dataset processing parameters.
        """
        self.config = config
        self._setup_paths()
        self._create_directories()
        
        # Number of worker processes for parallel execution
        self.num_workers = os.cpu_count() or 4
        
        # Precompile regex patterns for better performance
        self.row_col_pattern = re.compile(r"Row\(\d+\)_Col\(\d+\)")
        
    def _setup_paths(self):
        """Set up all necessary paths from the configuration."""
        # Base directories
        self.raw_dir = self.config['dataset_path']['raw']
        self.processed_dir = self.config['dataset_path']['processed']
        self.gen_dir = self.config['dataset_path']['gen']
        
        # File lists for dataset splitting
        self.train_file_path = os.path.join(self.raw_dir, 'train.txt')
        self.val_file_path = os.path.join(self.raw_dir, 'val.txt')
        self.test_file_path = os.path.join(self.raw_dir, 'test.txt')
        
        # Input directories for images and masks
        self.input_img_dir = os.path.join(self.raw_dir, 'TrueOrtho_resample')
        self.input_mask_dir = os.path.join(self.raw_dir, 'Mask2_resample')
        
        # Temporary directories for generated patches
        self.images_dir = os.path.join(self.gen_dir, "images")
        self.labels_dir = os.path.join(self.gen_dir, "labels")
        
        # Patch configuration
        self.patch_size = self.config['patch']['size']
        self.patch_extension = self.config['patch']['extension']
        self.zero_threshold = self.config['patch']['zero_threshold']
        
        # Split configuration
        self.split_seed = self.config['patch']['split']['seed']
        self.split_ratio = self.config['patch']['split']['ratio']
        
    def _create_directories(self):
        """Create all necessary directories."""
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
    def process_dataset(self, mode="file"):
        """
        Process the dataset based on the provided configuration.
        
        Args:
            mode (str): Split mode - "auto" for automatic split or "file" for predefined split.
        """
        try:
            # Step 1: Generate patches from raw images and masks
            self._process_all_images_parallel()
            
            # Step 2: Split the dataset according to the specified mode
            if mode == "auto":
                self._split_dataset_auto()
            else:
                self._split_dataset_file()
                
            # Step 3: Clean up temporary files
            self._cleanup()
            
            print(f"Dataset processing completed successfully. Output in {self.processed_dir}")
            
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
    
    def _process_image(self, filename):
        """
        Process a single image and its mask to generate patches.
        
        Args:
            filename: The filename of the image to process
            
        Returns:
            int: Number of patches generated
        """
        if not filename.endswith(self.patch_extension):
            return 0
            
        img_path = os.path.join(self.input_img_dir, filename)
        mask_path = os.path.join(self.input_mask_dir, f"Mask{filename[5:]}")
        
        # Skip if files don't exist
        if not (os.path.exists(img_path) and os.path.exists(mask_path)):
            return 0
        
        # Use more efficient image reading
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        
        if mask is None:
            return 0
            
        original_name = os.path.splitext(filename)[0]
        return self._crop_and_patch(image, mask, original_name)
            
    def _process_all_images_parallel(self):
        """Process all images and masks in parallel to generate patches."""
        filenames = [f for f in os.listdir(self.input_img_dir) if f.endswith(self.patch_extension)]
        
        patch_count = 0
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._process_image, filename) for filename in filenames]
            
            # Use tqdm for progress tracking
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                patch_count += future.result()
                
        print(f"Generated {patch_count} patches from {len(filenames)} images")
    
    def _crop_and_patch(self, image, mask, original_name):
        """
        Crop images to patch size multiples and create patches.
        
        Args:
            image: The input image
            mask: The corresponding mask
            original_name: The base name to use for the generated patches
            
        Returns:
            int: Number of patches generated
        """
        # Use faster numpy operations for cropping
        height, width = image.shape[:2]
        new_height = height - (height % self.patch_size)
        new_width = width - (width % self.patch_size)
        
        if new_height == 0 or new_width == 0:
            return 0
            
        # Crop using efficient slicing
        cropped_image = image[:new_height, :new_width]
        cropped_mask = mask[:new_height, :new_width]
        
        # Generate patches
        patches_img = patchify(cropped_image, (self.patch_size, self.patch_size, 3), step=self.patch_size)
        patches_mask = patchify(cropped_mask, (self.patch_size, self.patch_size, 3), step=self.patch_size)
        
        # Use vectorized operations where possible
        patches_saved = 0
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, 0]
                single_patch_mask = patches_mask[i, j, 0]
                
                # Filter patches with too many zeros - vectorized operation
                if self._is_patch_valid(single_patch_mask):
                    patch_name = f"{original_name}_patch_{i}_{j}{self.patch_extension}"
                    cv2.imwrite(os.path.join(self.images_dir, patch_name), single_patch_img)
                    cv2.imwrite(os.path.join(self.labels_dir, patch_name), single_patch_mask)
                    patches_saved += 1
                    
        return patches_saved
    
    def _is_patch_valid(self, patch_mask):
        """
        Check if a patch is valid based on the zero pixel ratio using vectorized operations.
        
        Args:
            patch_mask: The mask patch to check
            
        Returns:
            bool: True if the patch is valid, False otherwise
        """
        # Vectorized operation to count zeros
        zero_pixels = np.sum(np.all(patch_mask == 0, axis=2))
        total_pixels = patch_mask.shape[0] * patch_mask.shape[1]
        zero_ratio = zero_pixels / total_pixels
        
        return zero_ratio < self.zero_threshold
    
    def _split_dataset_auto(self):
        """Split the dataset automatically using splitfolders."""
        splitfolders.ratio(
            self.gen_dir,
            self.processed_dir,
            seed=self.split_seed,
            ratio=self.split_ratio
        )
    
    def _parse_file_list(self, file_path):
        """
        Parse a file list and extract pattern identifiers.
        
        Args:
            file_path: Path to the file containing the list
            
        Returns:
            set: Set of pattern identifiers for faster lookups
        """
        try:
            with open(file_path, 'r') as f:
                file_list = f.read().splitlines()
            
            # Use a set for O(1) lookups
            patterns = set()
            for filename in file_list:
                match = self.row_col_pattern.search(filename)
                if match:
                    patterns.add(match.group())
            
            return patterns
        except Exception as e:
            print(f"Error parsing file {file_path}: {str(e)}")
            return set()
    
    def _split_dataset_file(self):
        """Split the dataset according to predefined file lists."""
        # Create output directory structure
        sub_dirs = ['train', 'val', 'test']
        sub_folders = ['images', 'labels']
        
        os.makedirs(self.processed_dir, exist_ok=True)
        
        for sub_dir in sub_dirs:
            for sub_folder in sub_folders:
                os.makedirs(os.path.join(self.processed_dir, sub_dir, sub_folder), exist_ok=True)
        
        # Get file lists for each split - using sets for faster lookups
        train_images = self._parse_file_list(self.train_file_path)
        val_images = self._parse_file_list(self.val_file_path)
        test_images = self._parse_file_list(self.test_file_path)
        
        # Move files to appropriate directories in parallel
        self._move_files_to_splits_parallel(train_images, val_images, test_images)
    
    def _get_destination_dir(self, filename, subdir, train_images, val_images, test_images):
        """
        Determine the destination directory for a given file.
        
        Args:
            filename: The filename to check
            subdir: The subdirectory (images or labels)
            train_images: Set of pattern identifiers for training set
            val_images: Set of pattern identifiers for validation set
            test_images: Set of pattern identifiers for test set
            
        Returns:
            str: Path to the destination directory or None if no match
        """
        match = self.row_col_pattern.search(filename)
        if not match:
            return None
        
        pattern = match.group()
        
        if pattern in train_images:
            return os.path.join(self.processed_dir, 'train', subdir)
        elif pattern in val_images:
            return os.path.join(self.processed_dir, 'val', subdir)
        elif pattern in test_images:
            return os.path.join(self.processed_dir, 'test', subdir)
        else:
            return None
    
    def _copy_file(self, args):
        """
        Copy a file to its destination directory.
        
        Args:
            args: Tuple containing (source_path, dest_dir, filename)
            
        Returns:
            bool: True if copy was successful
        """
        source_path, dest_dir, filename = args
        try:
            shutil.copy2(source_path, os.path.join(dest_dir, filename))
            return True
        except Exception as e:
            print(f"Error copying {filename}: {str(e)}")
            return False
    
    def _move_files_to_splits_parallel(self, train_images, val_images, test_images):
        """
        Move files to their respective split directories in parallel.
        
        Args:
            train_images: Set of pattern identifiers for training set
            val_images: Set of pattern identifiers for validation set
            test_images: Set of pattern identifiers for test set
        """
        # Process each subdirectory in parallel
        for subdir in ['images', 'labels']:
            source_dir = os.path.join(self.gen_dir, subdir)
            filenames = os.listdir(source_dir)
            
            # Prepare arguments for parallel execution
            copy_tasks = []
            for filename in filenames:
                dest_dir = self._get_destination_dir(filename, subdir, train_images, val_images, test_images)
                if dest_dir:
                    source_path = os.path.join(source_dir, filename)
                    copy_tasks.append((source_path, dest_dir, filename))
            
            # Copy files in parallel
            success_count = 0
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._copy_file, args) for args in copy_tasks]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Copying {subdir}"):
                    if future.result():
                        success_count += 1
            
            print(f"Successfully copied {success_count} of {len(copy_tasks)} {subdir}")
    
    def _cleanup(self):
        """Clean up temporary files and directories."""
        if os.path.exists(self.gen_dir):
            shutil.rmtree(self.gen_dir)
            
# Usage example 
def process_dataset(process_config, mode="file"):
    """
    Process the dataset based on the provided configuration.

    Args:
        process_config (dict): Configuration dictionary containing dataset processing parameters.
        mode (str): Split mode - "auto" for automatic split or "file" for predefined split.
    """
    processor = DatasetProcessor(process_config)
    processor.process_dataset(mode)
