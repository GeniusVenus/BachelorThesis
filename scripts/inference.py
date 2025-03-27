import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.argument import parse_inference_args
from src.utils.config import load_config_from_inference_args
from src.models import SegmentationModel
from src.utils.checkpoint import load_checkpoint
from src.inference.predictor import Predictor

# Import ClearML
from clearml import Task

os.environ['CLEARML_CONFIG_FILE'] = '../clearml.conf'

def setup_clearml_inference(args, config):
    """Setup ClearML task for inference tracking"""
    # Create a task
    if config['model']['params']['encoder_name']:
        task_name = f"{config['loss']['name']}_{config['model']['params']['encoder_name']}_{config['model']['name']}"
    else:
        task_name = f"{config['loss']['name']}_{config['model']['params']['pretrained_model']}_{config['model']['name']}"


    project_name = config.get('clearml', {}).get('project_name', 'Segmentation')

    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        task_type="Inference",
    )

    # Connect configuration to the task
    task.connect_configuration(config)

    # Log the command line arguments
    task.connect(vars(args))

    return task

def main():
    # Parse command line arguments
    args = parse_inference_args()

    print("==============Arguments==============")
    print(args)

    # Load and merge configurations
    config = load_config_from_inference_args(args)

    print("==============Configs==============")
    print(config)

    # Setup ClearML for inference
    task = setup_clearml_inference(args, config)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = SegmentationModel.get_model(config['model']['name'], **config['model'].get('params', {}))

    input_path = config['inference']['input_path']
    output_path = config['inference']['output_path']

    for checkpoint in config['inference']['checkpoints']:
        load_checkpoint(model, checkpoint)

        # Log the checkpoint being used
        task.logger.report_text(f"Using checkpoint: {checkpoint}")

        # Initialize predictor
        predictor = Predictor(model=model, config=config, device=device, task=task)

        predictor.predict_large_image(input_path, output_path)

if __name__ == '__main__':
    main()
