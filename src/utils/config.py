import yaml
import os

CONFIG_FOLDER = 'configs'
CHECKPOINT_FOLDER = 'checkpoints'
AUGMENTATIONS_CONFIG_FOLDER = 'configs/augmentations'
DATA_CONFIG_FOLDER = 'configs/data'
PROCESS_CONFIG_FOLDER = 'configs/process'
EXPERIMENTS_CONFIG_FOLDER = 'configs/experiments'
LOSSES_CONFIG_FOLDER = 'configs/losses'
MODELS_CONFIG_FOLDER = 'configs/models'
LOSSES = ['cross_entropy','focal', 'dice', 'tversky', 'jaccard', 'lovasz']

def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def load_config_from_training_args(args):
    # Load base config.py
    base_config = load_config(f"{os.path.join(CONFIG_FOLDER, args.base_config)}.yml")

    if args.config:
        # Load experiment config.py
        exp_config = load_config(f"{os.path.join(CONFIG_FOLDER, args.config)}.yml")
        # Merge configs
        config = merge_configs(base_config, exp_config)
    else:
        config = base_config

    # Override with command line arguments
    if args.model:
        config = merge_configs(config, load_config(f"{os.path.join(MODELS_CONFIG_FOLDER, args.model)}.yml"))

    if args.backbone:
        if 'encoder_name' in config['model']['params'] and config['model']['params']['encoder_name']:
            config['model']['params']['encoder_name'] = args.backbone
        else:
            config['model']['params']['pretrained_model'] = args.backbone

    if args.loss:
        config = merge_configs(config, load_config(f"{os.path.join(LOSSES_CONFIG_FOLDER, args.loss)}.yml"))

    if args.dataset:
        data_config = load_config(f"{os.path.join(DATA_CONFIG_FOLDER, args.dataset)}.yml")
        config = merge_configs(config, data_config)
        config['model']['params']['classes'] = data_config['data']['num_classes']
        config['metrics']['num_classes'] = data_config['data']['num_classes']

    if args.batch_size:
        config['data']['batch_size'] = args.batch_size

    if args.learning_rate:
        config['optimizer']['learning_rate'] = args.learning_rate

    if args.epochs:
        config['training']['epochs'] = args.epochs

    if args.checkpoint:
        config['training']['model_checkpoint'] = f"checkpoints/{args.checkpoint}.pth"

    return config

def load_config_from_inference_args(args):
    # Load base config.py
    base_config = load_config(f"{os.path.join(CONFIG_FOLDER, args.base_config)}.yml")

    if args.config:
        # Load experiment config.py
        exp_config = load_config(f"{os.path.join(CONFIG_FOLDER, args.config)}.yml")
        # Merge configs
        config = merge_configs(base_config, exp_config)
    else:
        config = base_config

    # Override with command line arguments
    if args.model:
        config = merge_configs(config, load_config(f"{os.path.join(MODELS_CONFIG_FOLDER, args.model)}.yml"))

    if args.backbone:
        if 'encoder_name' in config['model']['params'] and config['model']['params']['encoder_name']:
            config['model']['params']['encoder_name'] = args.backbone
        else:
            config['model']['params']['pretrained_model'] = args.backbone

    if args.loss:
        config['inference']['checkpoints'] = []

        if args.loss == 'all' or config['loss'] == {}:
            config['loss'] = {}
            loss_list = LOSSES
        else:
            config = merge_configs(config, load_config(f"{os.path.join(LOSSES_CONFIG_FOLDER, args.loss)}.yml"))
            loss_list = [args.loss]

        # Generate checkpoint paths based on model configuration
        for loss_name in loss_list:
            if 'pretrained_model' in config['model']['params']:
                backbone = config['model']['params']['pretrained_model']
            else:
                backbone = config['model']['params']['encoder_name']
            checkpoint_path = f"{loss_name}_{backbone}_{config['model']['name']}.pth"
            config['inference']['checkpoints'].append(os.path.join(CHECKPOINT_FOLDER, checkpoint_path))

    if args.checkpoint:
        config['inference']['checkpoints'] = [os.path.join(CHECKPOINT_FOLDER, f"{args.checkpoint}.pth")]

    if args.dataset:
        data_config = load_config(f"{os.path.join(DATA_CONFIG_FOLDER, args.dataset)}.yml")
        config = merge_configs(config, data_config)
        config['model']['params']['classes'] = data_config['data']['num_classes']

    return config

def load_config_from_evaluation_args(args):
    # Load base config.py
    base_config = load_config(f"{os.path.join(CONFIG_FOLDER, args.base_config)}.yml")

    if args.config:
        # Load experiment config.py
        exp_config = load_config(f"{os.path.join(CONFIG_FOLDER, args.config)}.yml")
        # Merge configs
        config = merge_configs(base_config, exp_config)
    else:
        config = base_config

    # Override with command line arguments
    if args.model:
        config = merge_configs(config, load_config(f"{os.path.join(MODELS_CONFIG_FOLDER, args.model)}.yml"))

    if args.backbone:
        if 'encoder_name' in config['model']['params'] and config['model']['params']['encoder_name']:
            config['model']['params']['encoder_name'] = args.backbone
        else:
            config['model']['params']['pretrained_model'] = args.backbone

    if args.loss:
        config['inference']['checkpoints'] = []

        if args.loss == 'all' or config['loss'] == {}:
            config['loss'] = {}
            loss_list = LOSSES
        else:
            config = merge_configs(config, load_config(f"{os.path.join(LOSSES_CONFIG_FOLDER, args.loss)}.yml"))
            loss_list = [args.loss]

        # Generate checkpoint paths based on model configuration
        for loss_name in loss_list:
            if 'pretrained_model' in config['model']['params']:
                backbone = config['model']['params']['pretrained_model']
            else:
                backbone = config['model']['params']['encoder_name']
            checkpoint_path = f"{loss_name}_{backbone}_{config['model']['name']}.pth"
            config['evaluation']['checkpoints'].append(checkpoint_path)

    if args.dataset:
        data_config = load_config(f"{os.path.join(DATA_CONFIG_FOLDER, args.dataset)}.yml")
        config = merge_configs(config, data_config)
        config['model']['params']['classes'] = data_config['data']['num_classes']

    return config

def load_config_from_process_args(args):
    # Load base config.py
    base_config = load_config(f"{os.path.join(PROCESS_CONFIG_FOLDER, args.base_config)}.yml")

    if args.config:
        # Load experiment config.py
        exp_config = load_config(f"{os.path.join(PROCESS_CONFIG_FOLDER, args.config)}.yml")
        # Merge configs
        config = merge_configs(base_config, exp_config)
    else:
        config = base_config

    if args.raw_dir:
        config['dataset_path']['raw'] = args.raw_dir
    if args.processed_dir:
        config['dataset_path']['processed'] = args.processed_dir
    if args.patch_size:
        config['patch']['size'] = args.patch_size
    if args.zero_threshold:
        config['patch']['zero_threshold'] = args.zero_threshold
    if args.split_ratio:
        config['patch']['split']['ratio'] = args.split_ratio
    if args.split_seed:
        config['patch']['split']['seed'] = args.split_seed

    return config

def merge_configs(root_config, new_config):
    for key in new_config:
        root_config[key] = new_config[key]
    return root_config

