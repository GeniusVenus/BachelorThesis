import os
import cv2
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.data.preprocessing.base_processor import BaseDatasetProcessor


class OpenEarthMapProcessor(BaseDatasetProcessor):
    """A class to handle processing of OpenEarthMap datasets with images and labels folders."""

    def __init__(self, config):
        """
        Initialize the OpenEarthMapProcessor with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing dataset processing parameters.
        """
        super().__init__(config)

    def _process_images(self):
        """Process all datasets in the raw directory."""
        # Find all dataset folders
        dataset_folders = [f for f in os.listdir(self.raw_dir)
                          if os.path.isdir(os.path.join(self.raw_dir, f))]

        print(f"Found {len(dataset_folders)} dataset folders to process")

        # Process each dataset folder
        total_patches = 0
        for folder in dataset_folders:
            folder_path = os.path.join(self.raw_dir, folder)
            patches = self._process_dataset_folder(folder_path, folder)
            total_patches += patches

        print(f"Generated {total_patches} patches from {len(dataset_folders)} dataset folders")

    def _process_dataset_folder(self, folder_path, folder_name):
        """
        Process a single dataset folder containing images and labels subfolders.

        Args:
            folder_path: Path to the dataset folder
            folder_name: Name of the dataset folder for naming patches

        Returns:
            int: Number of patches generated
        """
        images_folder = os.path.join(folder_path, "images")
        labels_folder = os.path.join(folder_path, "labels")

        if not (os.path.exists(images_folder) and os.path.exists(labels_folder)):
            print(f"Skipping {folder_name}: missing images or labels folder")
            return 0

        # Get all image files
        image_files = glob.glob(os.path.join(images_folder, f"*{self.patch_extension}"))

        if not image_files:
            print(f"No images found in {folder_name}")
            return 0

        print(f"Processing {len(image_files)} images from {folder_name}")

        # Process images in parallel
        patch_count = 0
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []

            for img_path in image_files:
                img_filename = os.path.basename(img_path)
                label_path = os.path.join(labels_folder, img_filename)

                if not os.path.exists(label_path):
                    continue

                futures.append(
                    executor.submit(
                        self._process_image_pair,
                        img_path,
                        label_path,
                        folder_name
                    )
                )

            # Use tqdm for progress tracking
            for future in tqdm(as_completed(futures), total=len(futures),
                              desc=f"Processing {folder_name}"):
                patch_count += future.result()

        print(f"Generated {patch_count} patches from {folder_name}")
        return patch_count

    def _process_image_pair(self, img_path, label_path, folder_name):
        """
        Process a single image and its corresponding label to generate patches.

        Args:
            img_path: Path to the image file
            label_path: Path to the label file
            folder_name: Name of the dataset folder for naming patches

        Returns:
            int: Number of patches generated
        """
        try:
            # Read images
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(label_path, cv2.IMREAD_COLOR)

            if image is None or mask is None:
                return 0

            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            original_name = f"{folder_name}_{base_name}"

            return self._crop_and_patch(image, mask, original_name)

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return 0


def process_openearthmap_dataset(process_config):
    """
    Process the OpenEarthMap dataset based on the provided configuration.

    Args:
        process_config (dict): Configuration dictionary containing dataset processing parameters.
    """
    processor = OpenEarthMapProcessor(process_config)
    processor.process_dataset()

