import os
import cv2
import glob
import shutil
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
                          if os.path.isdir(os.path.join(self.raw_dir, f))]  # Skip split folders if they exist

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
            original_name = f"{base_name}"

            return self._crop_and_patch(image, mask, original_name)

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return 0

    def _parse_file_list(self, file_path):
        """
        Parse a file list and extract filenames.

        Args:
            file_path: Path to the file containing the list

        Returns:
            set: Set of filenames for faster lookups
        """
        try:
            with open(file_path, 'r') as f:
                file_list = f.read().splitlines()

            # Use a set for O(1) lookups
            filenames = set()
            for line in file_list:
                # Extract just the filename without path or extension
                filename = os.path.splitext(os.path.basename(line))[0]
                filenames.add(filename)

            return filenames
        except Exception as e:
            print(f"Error parsing file {file_path}: {str(e)}")
            return set()

    def _get_destination_dir(self, filename, subdir, train_images, val_images, test_images):
        """
        Determine the destination directory for a given file.

        Args:
            filename: The filename to check
            subdir: The subdirectory (images or labels)
            train_images: Set of filenames for training set
            val_images: Set of filenames for validation set
            test_images: Set of filenames for test set

        Returns:
            str: Path to the destination directory or None if no match
        """
        # Extract the base name without patch information and folder prefix
        # Format is typically: folder_name_base_name_patch_i_j.extension
        parts = os.path.splitext(filename)[0].split('_patch_')[0]

        # Extract the base image name (without folder prefix)
        # This is what we need to match against the file lists
        if '_' in parts:
            # The first part is the folder name, the rest is the base image name
            folder_prefix, *base_parts = parts.split('_', 1)
            base_image_name = base_parts[0] if base_parts else ""
        else:
            base_image_name = parts

        # Check if this base name is in any of the split sets
        if base_image_name in train_images:
            return os.path.join(self.processed_dir, 'train', subdir)
        elif base_image_name in val_images:
            return os.path.join(self.processed_dir, 'val', subdir)
        elif base_image_name in test_images:
            return os.path.join(self.processed_dir, 'test', subdir)
        else:
            # If not found in any set, try to match any part of the filename
            # This is a fallback mechanism
            for train_img in train_images:
                if train_img in filename:
                    return os.path.join(self.processed_dir, 'train', subdir)

            for val_img in val_images:
                if val_img in filename:
                    return os.path.join(self.processed_dir, 'val', subdir)

            for test_img in test_images:
                if test_img in filename:
                    return os.path.join(self.processed_dir, 'test', subdir)

            # If still not found, default to train
            print(f"Warning: No match found for {filename}, defaulting to train set")
            return os.path.join(self.processed_dir, 'train', subdir)

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
            train_images: Set of filenames for training set
            val_images: Set of filenames for validation set
            test_images: Set of filenames for test set
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


def process_openearthmap_dataset(process_config, mode="file"):
    """
    Process the OpenEarthMap dataset based on the provided configuration.

    Args:
        process_config (dict): Configuration dictionary containing dataset processing parameters.
        mode (str): Split mode - "auto" for automatic split or "file" for predefined split.
    """
    processor = OpenEarthMapProcessor(process_config)
    processor.process_dataset(mode)

