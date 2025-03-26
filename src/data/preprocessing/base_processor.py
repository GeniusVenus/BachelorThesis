import os
import cv2
import numpy as np
from patchify import patchify
import splitfolders
import shutil

from src.data.preprocessing.open_earth_map_processor import process_openearthmap_dataset
from src.data.preprocessing.naver_processer import process_naver_dataset

class BaseDatasetProcessor:
    """Base class for dataset processing operations for image segmentation tasks."""

    def __init__(self, config):
        """
        Initialize the BaseDatasetProcessor with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing dataset processing parameters.
        """
        self.config = config
        self._setup_paths()
        self._create_directories()

        # Number of worker processes for parallel execution
        self.num_workers = os.cpu_count() or 4

    def _setup_paths(self):
        """Set up all necessary paths from the configuration."""
        # Base directories
        self.raw_dir = self.config['dataset_path']['raw']
        self.processed_dir = self.config['dataset_path']['processed']
        self.gen_dir = self.config['dataset_path']['gen']

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

    def process_dataset(self, mode="auto"):
        """
        Process the dataset based on the provided configuration.

        Args:
            mode (str): Split mode - "auto" for automatic split or "file" for predefined split.
        """
        try:
            # Step 1: Generate patches from raw images and masks
            self._process_images()

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

    def _process_images(self):
        """
        Process images to generate patches.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _process_images method")

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
        print(f"Splitting dataset with ratio {self.split_ratio}")

        splitfolders.ratio(
            self.gen_dir,
            self.processed_dir,
            seed=self.split_seed,
            ratio=self.split_ratio
        )

    def _split_dataset_file(self):
        """
        Split the dataset according to predefined file lists.
        This method should be implemented by subclasses that need file-based splitting.
        """
        raise NotImplementedError("Subclasses must implement _split_dataset_file method if needed")

    def _cleanup(self):
        """Clean up temporary files and directories."""
        if os.path.exists(self.gen_dir):
            shutil.rmtree(self.gen_dir)
            print(f"Cleaned up temporary directory: {self.gen_dir}")


def process_dataset(config):
  if config.get('dataset_name', None) == 'open_earth_map':
    process_openearthmap_dataset(config)
  elif config.get('dataset_name', None) == 'naver':
    process_naver_dataset(config)
  else:
    raise ValueError(f"Dataset name {config.get('dataset_name', None)} not supported")
