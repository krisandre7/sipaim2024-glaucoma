import inspect
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, validator
from src.bootstrap.registry import (MODELS, SCHEDULERS,
                                    LOSSES, OPTIMIZERS, DATAMODULES, METRICS)
import albumentations as A

# class DatasetConfig(BaseModel):
#     name: str

class DataModuleConfig(BaseModel):
    name: str
    args: dict


class TrainerConfig(BaseModel):
    name: Optional[str] = None
    args: dict

    # class TrainerArgs(BaseModel):
    #     max_epochs: int


class OptimizerConfig(BaseModel):
    name: str
    args: dict

    # class OptimizerArgs(BaseModel):
    #     lr: float


class ModelConfig(BaseModel):
    name: str
    args: dict


class Transform(BaseModel):
    name: str
    args: dict


class SchedulerConfig(BaseModel):
    name: str
    args: Optional[dict] = None


class MonitorConfig(BaseModel):
    project: str
    entity: Optional[str] = None
    job_name: Optional[str] = None
    mode: Literal['offline', 'online', 'disables'] = 'online'
    dir: Optional[str] = None
    id: Optional[str] = None
    resume: Optional[str] = 'allow'


class MetricsConfig(BaseModel):
    train: Optional[list[Union[str, dict[str, dict]]]] = None
    val: Optional[list[Union[str, dict[str, dict]]]] = None


class Config(BaseModel):
    # dataset: DatasetConfig
    datamodule: DataModuleConfig
    model: ModelConfig
    transforms: List[Transform] = None
    scheduler: Optional[SchedulerConfig] = None
    monitor: MonitorConfig
    optimizer: OptimizerConfig
    criterion: Union[str, dict]
    metrics: MetricsConfig
    seed: int
    trainer: TrainerConfig
    resume_path: Optional[str] = None

    # @validator("dataset")
    # def validate_dataset(cls, v, values):
    #     if v.name in DATASETS:
    #         return v
    #     else:
    #         raise ValueError(f"Dataset {v.name} not supported. Add it to ./src/bootstrap/registry.py")

    @validator("datamodule")
    def validate_datamodule(cls, v, values):
        if v.name in DATAMODULES:
            return v
        else:
            raise ValueError(f"Datamodule {v.name} not supported. Add it to ./src/bootstrap/registry.py")
        

    @validator("model")
    def validate_model(cls, v, values):
        if v.name in MODELS:
            return v
        else:
            raise ValueError(f"Model {v.name} not supported. Add it to ./src/bootstrap/registry.py")

    @validator("scheduler")
    def validate_scheduler(cls, v, values):
        if v.name in SCHEDULERS:
            return v
        else:
            raise ValueError(f"Scheduler {v.name} not supported. Add it to ./src/bootstrap/registry.py")

    @validator("optimizer")
    def validate_optimizer(cls, v, values):
        if v.name in OPTIMIZERS:
            return v
        else:
            raise ValueError(f"Optimizer {v.name} not supported. Add it to ./src/bootstrap/registry.py")

    @validator("criterion")
    def validate_criterion(cls, v, values):
        if isinstance(v, str):
            cls_name = v
        elif isinstance(v, dict):
            cls_name = v['name']
        else:
            raise ValueError(f"Criterion needs to either be a string or a dict")
        
        if cls_name in LOSSES:
            return v
        else:
            raise ValueError(f"Criterion {v} not supported. Add it to ./src/bootstrap/registry.py")

    @validator("metrics")
    def validate_metrics(cls, v: MetricsConfig, values):
        not_supported = []
        for metric in v.train + v.val:
            if isinstance(metric, str):
                continue

            if list(metric.keys())[0] not in METRICS:
                not_supported.append(f"- {metric}")

        if len(not_supported) > 0:
            raise ValueError(
                f"The following metrics are not supported. Add them to ./src/bootstrap/registry.py:\n".join(
                    not_supported))

        return v
    
    @validator("transforms")
    def validate_transforms(cls, v):
        transform_names = [transform.name for transform in v if transform.name in ['Resize', 'Normalize']]
        
        if len(transform_names) != 2:
            raise ValueError('Transforms Resize and Normalize are required, in that order.')

        for transform in v:
            if not hasattr(A, transform.name):
                raise ValueError(f'Transform {transform.name} does not exist in albumentations.')
            
            class_has_init_args(getattr(A, transform.name), transform.args)
        return v
    
    @validator("trainer")
    def validate_trainer(cls, v, values):
        if not 'save_metric' in v.args:
            return v
        
        metric_name = v.args['save_metric']
        
        metric_names = ['val_loss']
        for item in values['metrics'].val:
            if isinstance(item, dict):
                key = next(iter(item.keys()))
                metric_names.append(key)
            else:
                metric_names.append(item)

        if metric_name not in metric_names:
            raise ValueError(f'Save metric {metric_name} not in validation metrics list. Add it to the config.')
        
        return v

def class_has_init_args(cls, init_args):
        init_signature = inspect.signature(cls.__init__)
        for arg_name, arg_value in init_args.items():
            if arg_name not in init_signature.parameters:
                raise ValueError(f"Argument '{arg_name}' not found in '{cls.__name__}.__init__'")


# Parses and validates JSON
def validate_config(yaml_data):
    config = Config.model_validate(yaml_data)
    print('config is valid!')
    return config
