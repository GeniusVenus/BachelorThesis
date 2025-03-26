import re
import cv2
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from patchify import patchify
from tqdm import tqdm

from src.data.preprocessing.base_processor import BaseDatasetProcessor


class NAVERDatasetProcessor(BaseDatasetProcessor):
    """A class to handle dataset processing operations for NAVER image segmentation tasks."""

    def __init__(self, config):
        """
        Initialize the NAVERDatasetProcessor with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing dataset processing parameters.
        """
        super().__init__(config)

        # Precompile regex patterns for better performance
        self.row_col_pattern = re.compile(r"Row\(\d+\)_Col\(\d+\)")

        # Input directories for images and masks
        self.input_img_dir = os.path.join(self.raw_dir, 'TrueOrtho_resample')
        self.input_mask_dir = os.path.join(self.raw_dir, 'Mask2_resample')

    def _process_images(self):
        """Process all images and masks in parallel to generate patches."""
        self._process_all_images_parallel()

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


def process_naver_dataset(process_config, mode="file"):
    """
    Process the NAVER dataset based on the provided configuration.

    Args:
        process_config (dict): Configuration dictionary containing dataset processing parameters.
        mode (str): Split mode - "auto" for automatic split or "file" for predefined split.
    """
    processor = NAVERDatasetProcessor(process_config)
    processor.process_dataset(mode)
