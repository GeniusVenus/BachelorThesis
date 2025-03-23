from pathlib import Path
from clearml import Task

def setup_clearml(args, config):
    if not args.offline:
        # Initialize ClearML task
        task_name = args.task_name or Path(args.config).stem
        task = Task.init(project_name=args.project_name,
                         task_name=task_name)

        # Connect configuration
        task.connect_configuration(config)
        return task
    return None