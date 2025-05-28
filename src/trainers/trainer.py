import os

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from typing import Dict, Any
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.metric import AverageMeter
from src.utils.log import setup_logger, log_hyperparameters, log_epoch, log_best_model


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            loss_fn: nn.Module,
            config: Dict[str, Any],
            metric_meters: Dict[str, AverageMeter],
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model = model
        self.config = config

        self.model_name = config['model']['name']
        self.model_type = config['model']['type']
        self.model_encoder = config['model']['params']['encoder_name'] if config['model']['params']['encoder_name'] is not None else config['model']['params']['pretrained_model']

        self.model_checkpoint = config['training']['model_checkpoint']

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Setup optimization components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Loss and metrics
        self.loss_name = config['loss']['name']
        self.loss_fn = loss_fn
        self.metrics = self._setup_metrics()

        self.metric_meters = metric_meters
        self.best_val_loss = float('inf')

        # Setup logger
        self.logger = setup_logger(
            log_folder=f"{self.project_root}/logs/{self.model_name}",
            log_folder_name=self.loss_name,
            json_log_filename=f"{self.model_encoder}_training_log.json",
            text_log_filename=f"{self.model_encoder}_training_log.txt"
        )

    def _log_hyperparameters(self):
        """Log all relevant hyperparameters from config"""
        hyperparams = {
            'model_type': self.model_type,
            'optimizer': self.config.get('optimizer', {}).get('name', 'adam'),
            'learning_rate': self.config.get('optimizer', {}).get('learning_rate', 1e-4),
            'scheduler': self.config.get('scheduler', {}).get('name', 'reduce_on_plateau'),
            'batch_size': self.config.get('data', {}).get('batch_size', 'not specified'),
            'epochs': self.config.get('training', {}).get('epochs', 50)
        }
        log_hyperparameters(self.logger, hyperparams)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        opt_config = self.config.get('optimizer', {})
        param_groups = self.model.parameters()

        optimizers = {
            'adam': torch.optim.Adam
        }

        optimizer_cls = optimizers.get(opt_config['name'], torch.optim.Adam)
        return optimizer_cls(
            param_groups,
            lr=opt_config.get('learning_rate', 1e-4),
        )

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_config = self.config.get('scheduler', {})
        name = scheduler_config.get('name', 'reduce_on_plateau').lower()

        schedulers = {
            'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
            'step': torch.optim.lr_scheduler.StepLR,
            'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau
        }

        scheduler_cls = schedulers.get(name)
        if scheduler_cls:
            return scheduler_cls(
                self.optimizer,
                **scheduler_config.get('params', {})
            )
        return None

    def _setup_metrics(self):
        """Setup evaluation metrics"""
        metrics_config = self.config.get('metrics', {})
        num_classes = metrics_config['num_classes']

        metrics = {
            'iou': torchmetrics.JaccardIndex(num_classes=num_classes, task=metrics_config['task'],
                                             average=metrics_config['average']).to(self.device),
            'dice': torchmetrics.Dice(num_classes=num_classes, average=metrics_config['average']).to(self.device),
            'accuracy': torchmetrics.Accuracy(num_classes=num_classes, task=metrics_config['task'],
                                              average=metrics_config['average']).to(self.device)
        }
        return metrics
    
    def _reset_metrics(self):
        for meter in self.metric_meters.values():
            meter.reset()

    def _update_metrics(self, phase: str, y_pred_mask, y, loss, n):
        dice_score = self.metrics['dice'](y_pred_mask, y)
        iou_score = self.metrics['iou'](y_pred_mask, y)
        acc_score = self.metrics['accuracy'](y_pred_mask, y)

        self.metric_meters[f'{phase}_loss'].update(loss, n)
        self.metric_meters[f'{phase}_dice'].update(dice_score.item(), n)
        self.metric_meters[f'{phase}_iou'].update(iou_score.item(), n)
        self.metric_meters[f'{phase}_acc'].update(acc_score.item(), n)

    def _save_best_model(self, epoch: int, val_loss: float, file_name: str, metrics: Dict[str, float]):
        """Save the best model based on validation loss"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint(file_name, epoch, metrics)
            print(f"New best model '{file_name}' saved at epoch {epoch} with val loss {val_loss}")
            log_best_model(
                self.logger,
                epoch=epoch,
                train_loss=metrics['train_loss'],
                train_acc=metrics['train_acc'],
                val_loss=metrics['val_loss'],
                val_acc=metrics['val_acc'],
                train_iou=metrics['train_iou'],
                val_iou=metrics['val_iou'],
                train_dice=metrics['train_dice'],
                val_dice=metrics['val_dice']
            )

    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, filename)

    def train_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        for batch_id, batch in enumerate(tqdm(self.train_loader, desc='Training')):
            n = batch['image'].shape[0]
            x = batch['image'].to(self.device).float()
            y = batch['mask'].to(self.device).long()

            self.optimizer.zero_grad()

            y_pred = self.model(x)
            if self.model_type == 'transformer' and hasattr(y_pred, 'logits'):
                output = y_pred.logits
            else:
                output = y_pred

            if self.model_type == 'transformer':
                output = F.interpolate(output, size=y.shape[1:], mode='bilinear', align_corners=False)
            loss = self.loss_fn(output, y)

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                y_pred_mask = torch.argmax(torch.softmax(output, dim=1), dim=1)
                assert y_pred_mask.shape == y.shape, f"Shape mismatch: {y_pred_mask.shape} vs {y.shape}"
                self._update_metrics('train', y_pred_mask, y, loss.item(), n)

    def validate_epoch(self):
        """Validate the model for one epoch"""
        self.model.eval()
        with torch.no_grad():
            for batch_id, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                n = batch['image'].shape[0]
                x = batch['image'].to(self.device).float()
                y = batch['mask'].to(self.device).long()

                y_pred = self.model(x)
                if self.model_type == 'transformer' and hasattr(y_pred, 'logits'):
                    output = y_pred.logits
                else:
                    output = y_pred
                if self.model_type == 'transformer':
                    output = F.interpolate(output, size=y.shape[1:], mode='bilinear', align_corners=False)
                loss = self.loss_fn(output, y)

                y_pred_mask = torch.argmax(torch.softmax(output, dim=1), dim=1)
                assert y_pred_mask.shape == y.shape, f"Shape mismatch: {y_pred_mask.shape} vs {y.shape}"
                self._update_metrics('val', y_pred_mask, y, loss.item(), n)

    def train(self):
        training_config = self.config.get('training', {})
        num_epochs = training_config.get('epochs', 50)
        for epoch in range(1, num_epochs + 1):
            self._reset_metrics()
            self.train_epoch()
            self.validate_epoch()

            train_loss = self.metric_meters['train_loss'].avg
            val_loss = self.metric_meters['val_loss'].avg
            metrics = {key: meter.avg for key, meter in self.metric_meters.items()}

            self.scheduler.step(val_loss)

            # Log epoch results
            log_epoch(
                self.logger,
                epoch=epoch,
                train_loss=metrics['train_loss'],
                train_acc=metrics['train_acc'],
                val_loss=metrics['val_loss'],
                val_acc=metrics['val_acc'],
                train_iou=metrics['train_iou'],
                val_iou=metrics['val_iou'],
                train_dice=metrics['train_dice'],
                val_dice=metrics['val_dice']
            )

            print(
                f'Epoch {epoch}/{num_epochs}, '
                f'train_loss = {metrics.get("train_loss", "N/A"):.6f}, '
                f'train_accuracy = {metrics.get("train_acc", "N/A"):.6f}, '
                f'train_IoU = {metrics.get("train_iou", "N/A"):.6f}, '
                f'train_dice = {metrics.get("train_dice", "N/A"):.6f}, '
                f'val_loss = {metrics.get("val_loss", "N/A"):.6f}, '
                f'val_accuracy = {metrics.get("val_acc", "N/A"):.6f}, '
                f'val_IoU = {metrics.get("val_iou", "N/A"):.6f}, '
                f'val_dice = {metrics.get("val_dice", "N/A"):.6f}'
            )

            self._save_best_model(epoch, val_loss, self.model_checkpoint, metrics)
