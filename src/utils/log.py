import logging
import json
import os

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'message': record.getMessage(),
            'epoch': getattr(record, 'epoch', None),
            'train_accuracy': getattr(record, 'train_accuracy', None),
            'val_accuracy': getattr(record, 'val_accuracy', None),
            'train_loss': getattr(record, 'train_loss', None),
            'val_loss': getattr(record, 'val_loss', None),
            'train_iou': getattr(record, 'train_iou', None),
            'val_iou': getattr(record, 'val_iou', None),
            'train_dice': getattr(record, 'train_dice', None),
            'val_dice': getattr(record, 'val_dice', None),
        }
        return log_record

class JsonFileHandler(logging.FileHandler):
    def emit(self, record):
        log_entry = self.format(record)

        if os.path.exists(self.baseFilename):
            with open(self.baseFilename, 'r') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []

        logs.append(log_entry)

        with open(self.baseFilename, 'w') as f:
            json.dump(logs, f, indent=4)

class PlainTextFormatter(logging.Formatter):
    def format(self, record):
        log_message = f"{self.formatTime(record, self.datefmt)} - {record.levelname} - {record.getMessage()}"

        additional_info = []
        if hasattr(record, 'epoch'):
            additional_info.append(f"epoch: {record.epoch}")
        if hasattr(record, 'train_accuracy'):
            additional_info.append(f"train_accuracy: {record.train_accuracy}")
        if hasattr(record, 'val_accuracy'):
            additional_info.append(f"val_accuracy: {record.val_accuracy}")
        if hasattr(record, 'train_loss'):
            additional_info.append(f"train_loss: {record.train_loss}")
        if hasattr(record, 'val_loss'):
            additional_info.append(f"val_loss: {record.val_loss}")
        if hasattr(record, 'train_iou'):
            additional_info.append(f"train_iou: {record.train_iou}")
        if hasattr(record, 'val_iou'):
            additional_info.append(f"val_iou: {record.val_iou}")
        if hasattr(record, 'train_dice'):
            additional_info.append(f"train_dice: {record.train_dice}")
        if hasattr(record, 'val_dice'):
            additional_info.append(f"val_dice: {record.val_dice}")

        if additional_info:
            log_message += " | " + " | ".join(additional_info)

        return log_message


def setup_logger(log_folder="../../logs", log_folder_name=None, json_log_filename="training_log.json", text_log_filename="training_log.txt"):
    logger = logging.getLogger()
    # Create the directory if it doesn't exist
    os.makedirs(log_folder, exist_ok=True)

    # If log folder name is provided, use it as subdirectory name
    if log_folder_name:
        log_folder = os.path.join(log_folder, log_folder_name)

    # Create the directory if it doesn't exist
    os.makedirs(log_folder, exist_ok=True)

    # Full file paths for both JSON and text log files
    json_log_filepath = os.path.join(log_folder, json_log_filename)
    text_log_filepath = os.path.join(log_folder, text_log_filename)

    # Clear the log files if they already exist
    if os.path.exists(json_log_filepath):
        os.remove(json_log_filepath)
    if os.path.exists(text_log_filepath):
        os.remove(text_log_filepath)

    # JSON log handler (uses custom JsonFileHandler for array logging)
    json_handler = JsonFileHandler(json_log_filepath)
    json_handler.setFormatter(JsonFormatter())

    # Plain text log handler
    text_handler = logging.FileHandler(text_log_filepath)
    text_handler.setFormatter(PlainTextFormatter())

    # Add handlers to the logger
    logger.addHandler(json_handler)
    logger.addHandler(text_handler)
    logger.setLevel(logging.INFO)

    return logger

def log_hyperparameters(logger, hyperparams):
    hyperparams_str = ', '.join(f'{key}: {value}' for key, value in hyperparams.items())
    logger.info(f'Hyperparameters: {hyperparams_str}')

def log_epoch(logger, epoch, train_loss, train_acc, val_loss, val_acc, train_iou, val_iou, train_dice, val_dice):
    logger.info('Epoch Results', extra={
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_iou': train_iou,
        'val_iou': val_iou,
        'train_dice': train_dice,
        'val_dice': val_dice,
    })

def log_best_model(logger, epoch, train_loss, train_acc, val_loss, val_acc, train_iou, val_iou, train_dice, val_dice):
    logger.info(f"Best model saved at epoch {epoch} with val_loss = {val_loss:.6f}", extra={
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_iou': train_iou,
        'val_iou': val_iou,
        'train_dice': train_dice,
        'val_dice': val_dice,
    })