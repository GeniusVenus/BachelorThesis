from src.data.preprocessing.transforms import get_transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from typing import Dict, Any, Tuple, List, Optional
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

class Predictor:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        task=None  # ClearML Task object
    ):
        """
        Initialize the Predictor.

        Args:
            model: The trained model to use for prediction
            config: Configuration dictionary
            device: Device to run the model on
            task: ClearML Task for tracking
        """
        self.model = model
        self.config = config
        self.device = device
        self.task = task  # Store ClearML task
        self.model.to(self.device)
        self.model.eval()

        # Extract model type for handling different model outputs
        self.model_type = config['model']['type']

        # Get patch size from config
        self.patch_size = config['data']['patch']['size']

        # Get class colors for visualization
        self.num_classes = config['model']['params']['classes']

        # Get color map from config
        self.color_map = config['data']['color_map']

        # Get transforms for preprocessing
        self.transforms = get_transforms(config.get('augmentation', {}), 'test')

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess the image for model input.

        Args:
            image: Input image (H, W, C) in BGR format

        Returns:
            Preprocessed image tensor (B, C, H, W)
        """
        transformed = self.transforms(image=image)
        image_tensor = transformed['image']

        # Ensure tensor has batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    def predict_patch(self, patch: torch.Tensor) -> np.ndarray:
        """
        Predict a single patch.

        Args:
            patch: Image patch tensor (B, C, H, W)

        Returns:
            Predicted mask for the patch
        """
        with torch.no_grad():
            output = self.model(patch)

            # Handle different model output formats
            if self.model_type == 'transformer':
                output = output.logits
                output = F.interpolate(output, size=(256, 256), mode='bilinear', align_corners=False)

            # Apply softmax and get class predictions
            probs = torch.softmax(output, dim=1)
            pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()

            return pred_mask

    def predict_large_image(self, image_path: str, save_dir: Optional[str] = None) -> np.ndarray:
        """
        Predict segmentation for a large image by splitting it into patches.

        Args:
            image_path: Path to the large image
            save_dir: Directory to save the prediction result

        Returns:
            Predicted segmentation mask for the entire image
        """
        # Read the large image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not read image {image_path}")

        # Convert BGR to RGB for processing
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Log the input image to ClearML if available
        if self.task:
            self.task.logger.report_image("Input", base_name,
                                         image=original_image, iteration=0)

        # Calculate size divisible by patch_size
        SIZE_X = (original_image.shape[1] // self.patch_size) * self.patch_size
        SIZE_Y = (original_image.shape[0] // self.patch_size) * self.patch_size

        # Crop image to nearest size divisible by patch_size
        large_image = Image.fromarray(original_image)
        input_img = large_image.crop((0, 0, SIZE_X, SIZE_Y))
        input_img = np.array(input_img)

        # Create patches
        patches_img = patchify(input_img, (self.patch_size, self.patch_size, 3), step=self.patch_size)
        patches_img = patches_img[:, :, 0, :, :, :]

        # Initialize array for predictions with correct shape for unpatchify
        patched_prediction = np.zeros((
            patches_img.shape[0],
            patches_img.shape[1],
            1,
            self.patch_size,
            self.patch_size,
            1
        ))
        

        # Process each patch
        for i in tqdm(range(patches_img.shape[0]), desc="Processing rows"):
            for j in range(patches_img.shape[1]):
                # Extract patch
                single_patch_img = patches_img[i, j, :, :, :]

                try:
                    # Preprocess patch
                    patch_tensor = self.preprocess_image(single_patch_img)

                    # Predict patch
                    pred_mask = self.predict_patch(patch_tensor)

                    # Reshape for unpatchify
                    pred_mask = pred_mask.reshape((1, self.patch_size, self.patch_size, 1))

                    patched_prediction[i, j, 0] = pred_mask
                except Exception as e:
                    print(f"Error processing patch ({i}, {j}): {e}")
                    # Use a default prediction (e.g., all zeros) for this patch
                    patched_prediction[i, j, 0] = np.zeros((self.patch_size, self.patch_size, 1))

        # Reconstruct full image
        result_mask = unpatchify(patched_prediction, (input_img.shape[0], input_img.shape[1], 1))
        result_mask = result_mask[:, :, 0].astype(np.uint8)  # Remove the channel dimension

        # Create a colored visualization
        colored_mask = self.create_colored_mask(result_mask)

        # Log the prediction to ClearML if available
        if self.task:
            self.task.logger.report_image("Prediction", base_name,
                                         image=colored_mask, iteration=0)

        # Save the result if save_dir is provided
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            # Save the raw prediction mask
            mask_path = os.path.join(save_dir, f"{base_name}_pred_mask.png")
            cv2.imwrite(mask_path, result_mask)

            # Save the colored visualization
            colored_path = os.path.join(save_dir, f"{base_name}_pred_colored.png")
            cv2.imwrite(colored_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))

            # Log the saved file paths to ClearML if available
            if self.task:
                self.task.logger.report_text(f"Saved mask to: {mask_path}")
                self.task.logger.report_text(f"Saved colored visualization to: {colored_path}")

                # Upload the files as artifacts
                self.task.upload_artifact(f"{base_name}_mask", mask_path)
                self.task.upload_artifact(f"{base_name}_colored", colored_path)

        return result_mask

    def create_colored_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Create a colored visualization of the segmentation mask.

        Args:
            mask: Segmentation mask with class indices

        Returns:
            Colored visualization of the mask
        """
        # Initialize colored mask with zeros
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        # Apply colors directly from the color map
        for class_idx, color in self.color_map.items():
            # Create a binary mask for the current class
            binary_mask = (mask == int(class_idx))

            # Apply the color to all pixels of this class
            # RGB format for visualization
            colored_mask[binary_mask, 0] = color[0]  # R
            colored_mask[binary_mask, 1] = color[1]  # G
            colored_mask[binary_mask, 2] = color[2]  # B

        # Log class distribution to ClearML if available
        if self.task:
            # Ensure all classes in the color map are included in the distribution
            class_counts = {}
            total_pixels = mask.size

            # Initialize all classes with zero count
            for class_idx in self.color_map.keys():
                class_idx_int = int(class_idx)
                class_counts[class_idx_int] = np.sum(mask == class_idx_int)

            # Calculate percentages
            class_percentages = {
                str(class_idx): (count / total_pixels) * 100
                for class_idx, count in class_counts.items()
            }

            # Ensure labels and values match and are in the same order
            labels = [str(i) for i in sorted(class_counts.keys())]
            detailed_labels = [f"Class {label}" for label in labels]
            values = [class_percentages[label] for label in labels]

            # Report class distribution as a bar chart
            try:
                self.task.logger.report_histogram(
                    "Class Distribution",
                    "Percentage of Pixels",
                    values=values,
                    labels=detailed_labels,
                    iteration=0,
                )

                # Also log as scalars for easier tracking
                for label, value in zip(labels, values):
                    self.task.logger.report_scalar(
                        "Class Percentages",
                        f"Class {label}",
                        value,
                        iteration=0
                    )
            except Exception as e:
                # Fallback to just logging text if histogram fails
                self.task.logger.report_text(
                    f"Class distribution: {', '.join(f'{l}: {v:.2f}%' for l, v in zip(detailed_labels, values))}"
                )
                print(f"Warning: Failed to report histogram: {str(e)}")

        return colored_mask

    def predict_directory(self, input_dir: str, output_dir: str, file_extension: str = '.tif'):
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save prediction results
            file_extension: File extension of images to process
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get list of files to process
        files = [f for f in os.listdir(input_dir) if f.endswith(file_extension)]

        # Log the number of files to process
        if self.task:
            self.task.logger.report_text(f"Processing {len(files)} images from {input_dir}")

        for i, filename in enumerate(tqdm(files, desc="Processing images")):
            image_path = os.path.join(input_dir, filename)
            try:
                self.predict_large_image(image_path, output_dir)

                # Log progress
                if self.task:
                    self.task.logger.report_scalar("Progress", "Files Processed", i+1, i+1)
            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}"
                print(error_msg)
                if self.task:
                    self.task.logger.report_text(error_msg)
