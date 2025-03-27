import argparse

def parse_training_args():
    parser = argparse.ArgumentParser(description='Training script for semantic segmentation')

    # Basic arguments
    parser.add_argument('--config', type=str,
                        help='Path to the experiment config.py file')
    parser.add_argument('--base-config', type=str, default='training/base_config',
                        help='Path to the base config.py file')

    # Override config.py parameters
    parser.add_argument('--model', type=str,
                        help='Model architecture to use (overrides config.py)')
    parser.add_argument('--backbone', type=str, help='Backbone architecture to use')
    parser.add_argument('--loss', type=str,
                        help='Loss function to use (overrides config.py)')
    parser.add_argument('--dataset', type=str,
                        help='Dataset to use (overrides config.py)')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size (overrides config.py)')
    parser.add_argument('--learning-rate', type=float,
                        help='Learning rate (overrides config.py)')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs (overrides config.py)')

    # Training specific
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume from')

    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint to save')

    # ClearML specific
    # parser.add_argument('--project-name', type=str, default='semantic_segmentation',
    #                     help='ClearML project name')
    # parser.add_argument('--task-name', type=str,
    #                     help='ClearML task name (defaults to config.py name if not specified)')
    # parser.add_argument('--offline', action='store_true',
    #                     help='Run without ClearML logging')

    return parser.parse_args()

def parse_evaluation_args():
    parser = argparse.ArgumentParser(description='Evaluation script for semantic segmentation')

    # Basic arguments
    parser.add_argument('--config', type=str,
                        help='Path to the experiment config.py file')
    parser.add_argument('--base-config', type=str, default='evaluation/base_config',
                        help='Path to the base config.py file')

    # Override config.py parameters
    parser.add_argument('--model', type=str, help='Model architecture to use')
    parser.add_argument('--loss', type=str, help='Loss function to use')
    parser.add_argument('--backbone', type=str, help='Model backbone to use')
    parser.add_argument('--dataset', type=str, help='Dataset to use')

    return parser.parse_args()


def parse_inference_args():
    parser = argparse.ArgumentParser(description='Inference script for semantic segmentation')

     # Basic arguments
    parser.add_argument('--config', type=str,
                        help='Path to the experiment config.py file')
    parser.add_argument('--base-config', type=str, default='inference/base_config',
                        help='Path to the base config.py file')

     # Override config.py parameters
    parser.add_argument('--model', type=str, help='Model architecture to use')
    parser.add_argument('--loss', type=str, help='Loss function to use')
    parser.add_argument('--backbone', type=str, help='Model backbone to use')
    parser.add_argument('--dataset', type=str, help='Dataset to use')
    parser.add_argument('--input-path', type=str, help='Path to the input image')

    return parser.parse_args()

def parse_process_args():
    parser = argparse.ArgumentParser(description='Dataset processing script')

    # Basic arguments
    parser.add_argument('--base-config', type=str, default='base_config',
                        help='Path to the base dataset processing configuration file')
    parser.add_argument('--config', type=str, help='Path to the dataset processing configuration file')

    # Override config.py parameters
    parser.add_argument('--raw_dir', type=str,
                        help='Path to the raw dataset directory (overrides config)')
    parser.add_argument('--processed_dir', type=str,
                        help='Path to the processed dataset directory (overrides config)')
    parser.add_argument('--patch_size', type=int,
                        help='Size of patches to extract (overrides config)')
    parser.add_argument('--zero_threshold', type=float,
                        help='Threshold for filtering patches with too many zero pixels (overrides config)')
    parser.add_argument('--split_ratio', type=float, nargs='+',
                        help='Train/val/test split ratio (overrides config)')
    parser.add_argument('--split_seed', type=int,
                        help='Random seed for dataset splitting (overrides config)')

    return parser.parse_args()
