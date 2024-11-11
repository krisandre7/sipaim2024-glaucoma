import os
os.environ["KERAS_BACKEND"] = "torch"

import argparse
from pathlib import Path

import torch
import torchvision

from src.infra import MetricMonitor

torchvision.disable_beta_transforms_warning()

from src.bootstrap.bootstrap import Bootstrap, bootstrap_metrics
from src.infra.monitor.monitor import WandbExperimentMonitor

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Raigs-glaucoma training")

    parser.add_argument("config_file", type=str, help="Path to YAML configuration file")
    parser.add_argument("--dry_run", action='store_true', required=False,
                        help="Test project with one train and test pass.")
    parser.add_argument("--model_path", default=None, type=str, help="Path to YAML configuration file")

    args = parser.parse_args()

    vars = Bootstrap(args.config_file)

    monitor_args = vars.monitor_args.model_dump()

    model_args = vars.model_args
    resume_path = vars.resume_path

    if args.model_path is not None:
        resume_path = args.model_path

    model = vars.model

    start_epoch = 1
    previous_state = None
    if resume_path is not None:
        previous_state = torch.load(resume_path)
        start_epoch = previous_state.get('epoch', 1) + 1
        
        model = vars.bootstap_model(previous_state['config']['model']['name'])
        model = model(**previous_state['config']['model']['args'])
        model.load_state_dict(previous_state['model_state_dict'])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        monitor_args['step'] = previous_state['step']
        monitor_args['tags'] = [Path(resume_path).stem]
        previous_id = previous_state.get('previous_id')
        if previous_id is not None:
            monitor_args['tags'].append(f"prev-{previous_id}")
        print(f"Resuming training of {Path(resume_path).stem} from epoch {start_epoch}")
    else:
        model = model(**model_args)

    if args.dry_run:
        monitor_args['mode'] = 'offline'

    monitor = WandbExperimentMonitor(config=vars.config, **monitor_args)

    criterion = vars.criterion

    # Need to delete name key
    optimizer_args = vars.optimizer_args
    optimizer = vars.optimizer(params=model.parameters(), **optimizer_args)

    if previous_state is not None:
        optimizer.load_state_dict(previous_state['optimizer_state_dict'])

    # check if vars has scheduler_args

    if hasattr(vars, 'scheduler_args') and vars.scheduler_args is not None:
        scheduler_args = vars.scheduler_args
        scheduler = vars.scheduler(optimizer=optimizer, **scheduler_args)
    else:
        scheduler = None

    datamodule_args = vars.datamodule_args
    datamodule = vars.datamodule(transforms=vars.transforms, seed=vars.config.seed,
                                 **datamodule_args)

    metrics_args = bootstrap_metrics(datamodule.task, datamodule.num_classes, datamodule.num_labels,
                                     vars.config.metrics.model_dump())
    metrics = MetricMonitor(task=datamodule.task, **metrics_args)

    trainer_args = vars.trainer_args

    if args.dry_run:
        trainer_args['dry_run'] = True
    trainer = vars.trainer(criterion, optimizer, datamodule,
                           monitor, metrics, vars.config, scheduler, **trainer_args)

    trainer.fit(model, datamodule=datamodule, start_epoch=start_epoch)
