from torchmetrics import Metric, MetricCollection, MetricTracker
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.bootstrap.registry import (MODELS, SCHEDULERS, LOSSES, OPTIMIZERS, DATAMODULES, DATASETS, METRICS)
from src.bootstrap.models import validate_config
from src.enums import Task
import random
import numpy as np
import torch
import pickle

from src.trainers.trainer import Trainer

class Bootstrap:

    def __init__(self, config_path: str):

        with open(config_path, 'r') as file:
            self.config = validate_config(yaml.load(file, yaml.FullLoader))
        config = pickle.loads(pickle.dumps(self.config))

        self.seed = config.seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)

        self.trainer_args = config.trainer.args
        # self.trainer = TRAINERS[config.trainer.name]
        self.trainer = Trainer

        self.bootstap_model(self.config.model.name)

        transform_list = [getattr(A, transform.name)(**transform.args) for transform in config.transforms]
        transform_list.append(ToTensorV2())
        self.transforms = A.Compose(transform_list)

        # self.dataset_args = config.dataset
        # self.dataset = DATASETS[self.dataset_args.name]
        self.datamodule_args = config.datamodule.args
        self.datamodule_args['dataset'] = DATASETS[self.datamodule_args['dataset']]
        self.datamodule = DATAMODULES[config.datamodule.name]

        if config.scheduler:
            self.scheduler_args = config.scheduler.args
            self.scheduler = SCHEDULERS[config.scheduler.name]

        self.monitor_args = config.monitor

        self.optimizer_args = config.optimizer.args
        self.optimizer = OPTIMIZERS[config.optimizer.name]

        if isinstance(config.criterion, str):
            self.criterion = LOSSES[config.criterion]()
        elif isinstance(config.criterion, dict):
            self.criterion = LOSSES[config.criterion['name']](**config.criterion['args'])

        self.resume_path = config.resume_path
        
    def bootstap_model(self, model_name: str):
        self.model_args = self.config.model.args
        self.model = MODELS[model_name]
        
        return self.model


def bootstrap_metrics(task: Task, num_classes: int, num_labels: int, 
                      metrics_config, threshold = None) -> dict[str,list[Metric]]:
    all_metrics = dict()
    for stage, config_metrics in metrics_config.items():
        stage_metrics = []
        for metric in config_metrics:
            if isinstance(metric, str):
                metric_cls = METRICS[metric]
                kwargs = {'task': task,
                          'num_classes': num_classes,
                          'num_labels': num_labels}
                if metric not in ['sens_at_spec'] and threshold is not None:
                    kwargs['threshold'] = threshold
                    
                stage_metrics.append(metric_cls(**kwargs))

            if isinstance(metric, dict):
                metric_name, args = list(metric.items())[0]
                metric_cls = METRICS[metric_name]
                kwargs = {'task': task,
                          'num_classes': num_classes,
                          'num_labels': num_labels}
                kwargs.update(args)
                if metric_name not in ['sens_at_spec'] and threshold is not None:
                    kwargs['threshold'] = threshold

                stage_metrics.append(metric_cls(**kwargs))

        all_metrics[f'{stage}_metrics'] = stage_metrics

    return all_metrics
