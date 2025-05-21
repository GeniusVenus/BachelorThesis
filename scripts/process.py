import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing.process import process_dataset
from src.utils.argument import parse_process_args
from src.utils.config import load_config_from_process_args


def main():
    # Parse command line arguments
    args = parse_process_args()

    print("==============Arguments==============")
    print(args)

    # Load and merge configurations
    config = load_config_from_process_args(args)

    print("==============Configs==============")
    print(config)

    # Print configuration
    print("==============Dataset Processing Configuration==============")
    print(f"Raw directory: {config['dataset_path']['raw']}")
    print(f"Processed directory: {config['dataset_path']['processed']}")
    print(f"Patch size: {config['patch']['size']}")
    print(f"Patch extension: {config['patch']['extension']}")
    print(f"Zero threshold: {config['patch']['zero_threshold']}")
    print(f"Split ratio: {config['patch']['split']['ratio']}")
    print(f"Split seed: {config['patch']['split']['seed']}")

    # Process the dataset
    print("\n==============Processing Dataset==============")
    process_dataset(config)

    print("\n==============Dataset Processing Complete==============")
    print(f"Processed dataset saved to: {config['dataset_path']['processed']}")
    print(f"Dataset structure:")
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(config['dataset_path']['processed'], split)
        if os.path.exists(split_dir):
            num_images = len(os.listdir(os.path.join(split_dir, 'images')))
            num_labels = len(os.listdir(os.path.join(split_dir, 'labels')))
            print(f"  {split}: {num_images} images, {num_labels} labels")

if __name__ == '__main__':
    main()
