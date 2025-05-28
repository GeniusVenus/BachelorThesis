import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.metric import AverageMeter
import torchmetrics

class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize the Evaluator.
        
        Args:
            model: The trained model to evaluate
            config: Configuration dictionary
            device: Device to run the model on
        """
        self.model = model
        self.config = config
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        # Extract model type for handling different model outputs
        self.model_type = config['model']['type']
        
        # Get number of classes
        self.num_classes = config['model']['params']['classes']
        
        # Initialize metrics
        self.metrics = self._setup_metrics()
        
        # Setup torchmetrics 
        self.torchmetrics = {
            # Overall metrics (macro average)
            'iou': torchmetrics.JaccardIndex(
                num_classes=self.num_classes, 
                task='multiclass',
                average='macro'
            ).to(self.device),
            'dice': torchmetrics.Dice(
                num_classes=self.num_classes, 
                average='macro'
            ).to(self.device),
            'accuracy': torchmetrics.Accuracy(
                task='multiclass', 
                num_classes=self.num_classes,
                average='macro'
            ).to(self.device),
            
            # Per-class metrics (no averaging - returns tensor of per-class values)
            'iou_per_class': torchmetrics.JaccardIndex(
                num_classes=self.num_classes, 
                task='multiclass',
                average=None
            ).to(self.device),
            'accuracy_per_class': torchmetrics.Accuracy(
                task='multiclass',
                num_classes=self.num_classes,
                average=None
            ).to(self.device)
        }
    
    def _setup_metrics(self) -> Dict[str, AverageMeter]:
        """
        Set up metrics for evaluation.
        
        Returns:
            Dictionary of metric names to AverageMeter instances
        """
        metrics = {
            'loss': AverageMeter(),
            'iou': AverageMeter(),
            'dice': AverageMeter()
        }
        
        # Add per-class IoU and Dice metrics
        for i in range(self.num_classes):
            metrics[f'iou_class_{i}'] = AverageMeter()
            metrics[f'dice_class_{i}'] = AverageMeter()
            
        return metrics
    
    def _reset_metrics(self):
        """Reset all metrics."""
        for meter in self.metrics.values():
            meter.reset()
        
        # Reset torchmetrics
        for metric in self.torchmetrics.values():
            metric.reset()
            
    def _calculate_per_class_dice(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate per-class Dice coefficient manually.
        
        Args:
            y_pred: Predicted segmentation masks [B, H, W]
            y_true: Ground truth segmentation masks [B, H, W]
            
        Returns:
            Tensor of per-class Dice coefficients [num_classes]
        """
        dice_scores = torch.zeros(self.num_classes, device=self.device)
        
        for class_idx in range(self.num_classes):
            # Create binary masks for current class
            pred_class = (y_pred == class_idx).float()
            true_class = (y_true == class_idx).float()
            
            # Calculate intersection and union
            intersection = (pred_class * true_class).sum()
            total = pred_class.sum() + true_class.sum()
            
            # Calculate Dice coefficient
            if total > 0:
                dice_scores[class_idx] = (2.0 * intersection) / total
            else:
                # If class doesn't exist in batch, set to NaN
                dice_scores[class_idx] = float('nan')
                
        return dice_scores
    
    def _update_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, loss: float, batch_size: int):
        """
        Update metrics with batch results.
        
        Args:
            y_pred: Predicted segmentation masks
            y_true: Ground truth segmentation masks
            loss: Loss value
            batch_size: Batch size
        """
        # Update loss
        self.metrics['loss'].update(loss, batch_size)
        
        # Update overall metrics
        iou_value = self.torchmetrics['iou'](y_pred, y_true)
        dice_value = self.torchmetrics['dice'](y_pred, y_true)
        
        self.metrics['iou'].update(iou_value.item(), batch_size)
        self.metrics['dice'].update(dice_value.item(), batch_size)
        
        # Get per-class values (returns tensor of shape [num_classes])
        per_class_iou = self.torchmetrics['iou_per_class'](y_pred, y_true)
        per_class_accuracy = self.torchmetrics['accuracy_per_class'](y_pred, y_true)
        
        # Calculate per-class dice manually
        per_class_dice = self._calculate_per_class_dice(y_pred, y_true)
        
        # Update individual class metrics
        for i in range(self.num_classes):
            # Handle potential NaN values (when class doesn't appear in batch)
            if not torch.isnan(per_class_iou[i]):
                self.metrics[f'iou_class_{i}'].update(per_class_iou[i].item(), batch_size)
            if not torch.isnan(per_class_dice[i]):
                self.metrics[f'dice_class_{i}'].update(per_class_dice[i].item(), batch_size)
            if not torch.isnan(per_class_accuracy[i]):
                self.metrics[f'accuracy_class_{i}'].update(per_class_accuracy[i].item(), batch_size)
    
    def evaluate_dataloader(self, dataloader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """
        Evaluate the model on a dataloader.
        
        Args:
            dataloader: DataLoader for evaluation
            criterion: Loss function
            
        Returns:
            Dictionary of metric names to values
        """
        # Reset metrics
        self._reset_metrics()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Evaluate on dataloader
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                n = batch['image'].shape[0]
                images = batch['image'].to(self.device).float()
                masks = batch['mask'].to(self.device).long()

                outputs = self.model(images)
                if self.model_type == 'transformer' and hasattr(outputs, 'logits'):
                    outputs = outputs.logits

                if self.model_type == 'transformer':
                    outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)

                loss = criterion(outputs, masks)

                # Apply softmax before argmax to match training
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                
                # Update metrics
                self._update_metrics(preds, masks, loss.item(), n)
        
        # Get final metric values
        results = {name: meter.avg for name, meter in self.metrics.items()}
        
       # Debug: Get the final torchmetrics values to compare
        final_torchmetrics_iou = self.torchmetrics['iou'].compute()
        final_per_class_iou = self.torchmetrics['iou_per_class'].compute()
        
        # Verify that mIoU equals mean of per-class IoUs
        manual_miou = sum(results[f'iou_class_{i}'] for i in range(self.num_classes)) / self.num_classes
        torchmetrics_manual_miou = torch.nanmean(final_per_class_iou).item()
        
        print(f"Manual mIoU calculation (from AverageMeters): {manual_miou:.4f}")
        print(f"TorchMetrics mIoU (macro): {results['iou']:.4f}")
        print(f"TorchMetrics mIoU (direct): {final_torchmetrics_iou:.4f}")
        print(f"TorchMetrics per-class mean: {torchmetrics_manual_miou:.4f}")
        print(f"Per-class IoU values (torchmetrics): {final_per_class_iou}")
        print(f"Per-class IoU values (our calculation): {[results[f'iou_class_{i}'] for i in range(self.num_classes)]}")
        
        return results
    
    def _log_metrics_to_clearml(self, results: Dict[str, float], split_name: str, iteration: int = 0):
        """
        Log metrics to ClearML.
        
        Args:
            results: Dictionary containing metric results
            split_name: Name of the data split
            iteration: Iteration number for logging
        """
        if not self.task:
            return

        # Get labels from config, defaulting to class indices if not found
        label_map = self.config['data'].get('labels', {})
        
        # Report overall metrics as text
        overall_metrics_text = (
            f"{split_name} Overall Metrics:\n"
            f"Loss: {results['loss']:.4f}\n"
            f"IoU: {results['iou']:.4f}\n"
            f"Dice: {results['dice']:.4f}\n"
            f"Accuracy: {results['accuracy']:.4f}"
        )
        self.task.logger.report_text(overall_metrics_text)

        # Report per-class metrics
        for i in range(self.num_classes):
            class_name = label_map.get(i, f"Class {i}")
            
            self.task.logger.report_histogram(
                title=f"{split_name}/Per-Class IoU",
                series=class_name,
                values=results[f'iou_class_{i}'],
                iteration=iteration,
                xaxis="Class",
                yaxis="IoU",
            )
            
            self.task.logger.report_histogram(
                title=f"{split_name}/Per-Class Dice",
                series=class_name,
                values=results[f'dice_class_{i}'],
                iteration=iteration,
                xaxis="Class",
                yaxis="Dice"
            )
            
            self.task.logger.report_histogram(
                title=f"{split_name}/Per-Class Accuracy",
                series=class_name,
                values=results[f'accuracy_class_{i}'],
                iteration=iteration, 
                xaxis="Class",
                yaxis="Accuracy"
            )

    def evaluate_split(self, split_dataloader: DataLoader, criterion: nn.Module, split_name: str = "val") -> Dict[str, float]:
        """
        Evaluate the model on a specific data split.
        
        Args:
            split_dataloader: DataLoader for the split
            criterion: Loss function
            split_name: Name of the split (e.g., "train", "val", "test")
            
        Returns:
            Dictionary of metric names to values
        """
        print(f"Evaluating on {split_name} split...")
        results = self.evaluate_dataloader(split_dataloader, criterion)
        
        # Print results
        print(f"{split_name.capitalize()} Loss: {results['loss']:.4f}")
        print(f"{split_name.capitalize()} IoU: {results['iou']:.4f}")
        print(f"{split_name.capitalize()} Dice: {results['dice']:.4f}")
        
        # Print per-class metrics
        print(f"\nPer-class metrics for {split_name} split:")
        for i in range(self.num_classes):
            print(f"Class {i}: IoU = {results[f'iou_class_{i}']:.4f}, Dice = {results[f'dice_class_{i}']:.4f}")
        
        return results
    
    def evaluate_all_splits(self, train_loader: DataLoader, val_loader: DataLoader, 
                           test_loader: Optional[DataLoader], criterion: nn.Module) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on all data splits.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data (optional)
            criterion: Loss function
            
        Returns:
            Dictionary of split names to metric dictionaries
        """
        results = {}
        
        # Evaluate on training data
        results['train'] = self.evaluate_split(train_loader, criterion, "train")
        
        # Evaluate on validation data
        results['val'] = self.evaluate_split(val_loader, criterion, "val")
        
        # Evaluate on test data if provided
        if test_loader is not None:
            results['test'] = self.evaluate_split(test_loader, criterion, "test")
        
        return results
