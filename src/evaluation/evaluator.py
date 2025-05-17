import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils.metric import AverageMeter
import torchmetrics
from clearml import Task

class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        task: Optional[Task] = None  # Add ClearML task parameter
    ):
        """
        Initialize the Evaluator.
        
        Args:
            model: The trained model to evaluate
            config: Configuration dictionary
            device: Device to run the model on
            task: ClearML Task for logging (optional)
        """
        self.model = model
        self.config = config
        self.device = device
        self.task = task  # Store ClearML task
        self.model.to(self.device)
        self.model.eval()
        
        # Extract model type for handling different model outputs
        self.model_type = config['model']['type']
        
        # Get number of classes
        self.num_classes = config['model']['params']['classes']
        
        # Initialize metrics
        self.metrics = self._setup_metrics()
        
        # Setup torchmetrics
        metrics_config = {'task': 'multiclass', 'average': 'macro'}
        self.torchmetrics = {
            'iou': torchmetrics.JaccardIndex(num_classes=self.num_classes, task=metrics_config['task'],
                                            average=metrics_config['average']).to(self.device),
            'dice': torchmetrics.Dice(num_classes=self.num_classes, average=metrics_config['average']).to(self.device),
            'accuracy': torchmetrics.Accuracy(task=metrics_config['task'], num_classes=self.num_classes,
                                            average=metrics_config['average']).to(self.device)
        }
        
        # Add per-class metrics
        for i in range(self.num_classes):
            self.torchmetrics[f'iou_class_{i}'] = torchmetrics.JaccardIndex(
                num_classes=self.num_classes, 
                task=metrics_config['task'],
                average=metrics_config['average']
            ).to(self.device)
            self.torchmetrics[f'dice_class_{i}'] = torchmetrics.Dice(
                num_classes=self.num_classes,
                average=metrics_config['average']
            ).to(self.device)
            self.torchmetrics[f'accuracy_class_{i}'] = torchmetrics.Accuracy(
                task=metrics_config['task'],
                num_classes=self.num_classes,
                average=metrics_config['average']
            ).to(self.device)
    
    def _setup_metrics(self) -> Dict[str, AverageMeter]:
        """
        Set up metrics for evaluation.
        
        Returns:
            Dictionary of metric names to AverageMeter instances
        """
        metrics = {
            'loss': AverageMeter(),
            'iou': AverageMeter(),
            'dice': AverageMeter(),
            'accuracy': AverageMeter()
        }
        
        # Add per-class metrics
        for i in range(self.num_classes):
            metrics[f'iou_class_{i}'] = AverageMeter()
            metrics[f'dice_class_{i}'] = AverageMeter()
            metrics[f'accuracy_class_{i}'] = AverageMeter()
            
        return metrics
    
    def _reset_metrics(self):
        """Reset all metrics."""
        for meter in self.metrics.values():
            meter.reset()
        
        # Reset torchmetrics
        for metric in self.torchmetrics.values():
            metric.reset()
    
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
        
        # Update torchmetrics
        iou_value = self.torchmetrics['iou'](y_pred, y_true)
        dice_value = self.torchmetrics['dice'](y_pred, y_true)
        accuracy_value = self.torchmetrics['accuracy'](y_pred, y_true)
        
        self.metrics['iou'].update(iou_value.item(), batch_size)
        self.metrics['dice'].update(dice_value.item(), batch_size)
        self.metrics['accuracy'].update(accuracy_value.item(), batch_size)
        
        # Update per-class metrics
        for i in range(self.num_classes):
            class_pred = (y_pred == i).long()
            class_true = (y_true == i).long()
            
            if class_true.sum() > 0 or class_pred.sum() > 0:
                class_iou = self.torchmetrics['iou'](class_pred, class_true)
                class_dice = self.torchmetrics['dice'](class_pred, class_true)
                class_accuracy = self.torchmetrics['accuracy'](class_pred, class_true)
                
                self.metrics[f'iou_class_{i}'].update(class_iou.item(), batch_size)
                self.metrics[f'dice_class_{i}'].update(class_dice.item(), batch_size)
                self.metrics[f'accuracy_class_{i}'].update(class_accuracy.item(), batch_size)
    
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
                # Get batch data
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device).long()  # Convert masks to Long type
                
                # Forward pass
                outputs = self.model(images)
                
                # Handle different model output formats
                if self.model_type == 'transformer':
                    outputs = outputs.logits
                
                # Calculate loss
                loss = criterion(outputs, masks)
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                
                # Update metrics
                self._update_metrics(preds, masks, loss.item(), images.size(0))
        
        # Get final metric values
        results = {name: meter.avg for name, meter in self.metrics.items()}
        
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
            
            # Report as individual scalars
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
        print(f"{split_name.capitalize()} Accuracy: {results['accuracy']:.4f}")
        print(f"{split_name.capitalize()} Loss: {results['loss']:.4f}")
        print(f"{split_name.capitalize()} IoU: {results['iou']:.4f}")
        print(f"{split_name.capitalize()} Dice: {results['dice']:.4f}")
        
        # Print per-class metrics
        print(f"\nPer-class metrics for {split_name} split:")
        for i in range(self.num_classes):
            print(f"Class {i}: IoU = {results[f'iou_class_{i}']:.4f}, "
                  f"Dice = {results[f'dice_class_{i}']:.4f}, "
                  f"Accuracy = {results[f'accuracy_class_{i}']:.4f}")
        
        # Log metrics to ClearML
        if self.task:
            self._log_metrics_to_clearml(results, split_name)
        
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
