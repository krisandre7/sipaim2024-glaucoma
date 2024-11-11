import json
from pathlib import Path
from typing import Optional, Union, Any
import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from torch import Tensor
import yaml
from src.bootstrap.registry import METRICS, MINIMZE_METRICS
from src.datamodules import DataModule
from tqdm import trange
from src import Stage
from src.enums import JustRAIGSTask, Task
from src.infra import MetricMonitor, WandbExperimentMonitor
from src.infra.monitor import WandbExperimentMonitor
import shutil

from src.utils import clear_model_files

class Trainer:

    def __init__(self,
                 criterion: nn.Module, optimizer: Optimizer, datamodule: DataModule,
                 monitor: WandbExperimentMonitor, metrics_monitor: MetricMonitor, 
                 config: dict, scheduler: nn.Module = None, save_metric: str = 'val_loss',
                 max_epochs: int = 1000, dry_run = False, log_step = 100, gpu_num = 0):
        if torch.cuda.is_available():
            device_name = f"cuda:{gpu_num}"
        else:
            device_name = "cpu"

        self.device = torch.device(device_name)
        print(f"Using {self.device} device")

        torch.cuda.empty_cache()

        self.config = config
        self.log_step = log_step
        self.dry_run = dry_run
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.monitor = monitor
        self.datamodule = datamodule
        self.max_epochs = max_epochs
        self.metrics_monitor = metrics_monitor
        
        self.save_metric_name = save_metric
        self.save_metric = save_metric
        self.maximize_metric = get_maximize_metric(self.save_metric_name)
        
        if not self.save_metric_name == 'val_loss':
            self.save_metric = get_metric_name(self.save_metric, self.datamodule.task)
        

    def validate(self, model, dataloader, epoch):
        model.eval()
        val_loss = 0

        with torch.no_grad():
            # with multiline_trange(len(dataloader)) as progress_bar:
            with trange(len(dataloader)) as progress_bar:
                for batch_idx, sample_batch in zip(progress_bar, dataloader):
                    X, y = sample_batch
                    X = X.to(self.device)
                    y = y.to(self.device)

                    # Forward Pass
                    output = model(X).squeeze()
                    
                    # Mask Nan values in y
                    nan_mask = ~torch.isnan(y)
                    valid_output = output[nan_mask]
                    valid_y = y[nan_mask].float()
                    
                    self.metrics_monitor.to_val(self.device)
                    self.metrics_monitor.update_val(output, y)
                    

                    # Compute Loss
                    loss = self.criterion(valid_output, valid_y)

                    log_dict = {'val_loss_step': loss.item()}

                    if batch_idx % self.log_step == 0:
                        self.monitor.run.log(log_dict)
                    val_loss += loss.item()

                    bar_postfix = f'Validate Loop: [epoch: {epoch:d}], iteration: {batch_idx+1:d}/{len(dataloader):d}, val_loss: {loss.item():.5}'
                    
                    # bar_postfix += self.metrics_monitor.val_metrics_str()
                    
                    progress_bar.set_description(bar_postfix)
                    
                    if self.dry_run:
                        break

        val_loss = val_loss / len(dataloader)

        if self.scheduler:
            self.scheduler.step(val_loss)

        return val_loss

    def train(self, model, dataloader, epoch):
        model.train()

        train_loss = 0
        with trange(len(dataloader)) as progress_bar:
            for batch_idx, sample_batch in zip(progress_bar, dataloader):

                X, y = sample_batch
                X = X.to(self.device)
                y = y.to(self.device)

                # Forward Pass
                output: Tensor = model(X).squeeze()
                
                if output.dim() == 1:
                    y = y.squeeze()
                    
                # Mask Nan values in y
                nan_mask = ~torch.isnan(y)
                
                
                valid_output = output[nan_mask]
                valid_y = y[nan_mask].float()
                
                self.metrics_monitor.to_train(self.device)
                self.metrics_monitor.update_train(output, y)
                
                
                # loss = self.criterion(output, y)
                loss = self.criterion(valid_output, valid_y)

                # Backward Pass
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                log_dict = {'train_loss_step': loss.item()}
                
                if batch_idx % self.log_step == 0:
                    self.monitor.run.log(log_dict)
                train_loss += loss.item()

                bar_postfix = f'Train Loop: [epoch: {epoch:d}], iteration: {batch_idx + 1:d}/{len(dataloader):d}, train_loss: {loss.item():.5}'
                
                # bar_postfix += self.metrics_monitor.train_metrics_str()
                
                progress_bar.set_description(bar_postfix)

                if self.dry_run:
                    break

        train_loss = train_loss / len(dataloader)

        return train_loss

    def fit(self, model: nn.Module, 
            train_dataloader: Union[Any, DataModule, None] = None,
            val_dataloader: Optional[Any] = None,
            datamodule: Optional[DataModule] = None,
            start_epoch=1):
        model = model.to(self.device)
        
        if isinstance(train_dataloader, DataModule):
            datamodule = train_dataloader
            train_dataloader = None
        
        # If you supply a datamodule you can't supply train_dataloader or val_dataloader
        if (train_dataloader is not None or val_dataloader is not None) and datamodule is not None:
            raise ValueError(
                "You cannot pass `train_dataloader` or `val_dataloader` to `trainer.fit(datamodule=...)`"
            )
        
        if datamodule is not None:
            datamodule.prepare_data()
            datamodule.setup(Stage.FIT)
            train_dataloader = datamodule.train_dataloader()
            val_dataloader = datamodule.val_dataloader()

        best_metric = 0 if self.maximize_metric else np.inf
        for epoch in range(start_epoch, self.max_epochs):
            self.metrics_monitor.reset()

            train_loss = self.train(model, train_dataloader, epoch)
            
            if hasattr(datamodule, 'negative_resample_step'):
                if datamodule.negative_resample_step is not None:
                    if epoch % datamodule.negative_resample_step == 0:
                        train_dataloader.dataset.resample_negatives()
            
            train_log_dict = {'train_loss': train_loss}
            train_log_dict.update(self.metrics_monitor.compute_train(plot_conf_matrix=True))
            self.monitor.run.log(train_log_dict)
            
            val_loss = self.validate(model, val_dataloader, epoch)
            
            val_log_dict = {'val_loss': val_loss, 'epoch': epoch}
            val_log_dict.update(self.metrics_monitor.compute_val(plot_conf_matrix=True))
            self.monitor.run.log(val_log_dict)
            
            if self.save_metric not in val_log_dict:
                raise ValueError(f'Save metric {self.save_metric} not in metrics.')
            
            metric = val_log_dict[self.save_metric]
            is_best = metric > best_metric if self.maximize_metric else metric < best_metric
            
            if is_best:
                best_metric = metric
                self.monitor.run.summary[f'best_{self.save_metric}'] = best_metric
                model.eval()
                
                config = self.config.model_dump()
                
                model_name = config['model']['args'].get('model_name')
                if model_name is None:
                    model_name = model.__module__.split('.')[-1]

                task = self.datamodule.task
                task = task.value if isinstance(self.datamodule.task, Task) else task
                if task == 'binary':
                    task = 'ref'
                elif task == 'multilabel':
                    task = 'just'
                
                run_dir = Path(self.monitor.run.dir)
                model_filename = f'{model_name}_{self.save_metric_name}_{best_metric:.4f}_{task}'
                model_path = run_dir / f'{model_filename}.pt'
                clear_model_files(run_dir)
                
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'model': model,
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'best_metric': best_metric,
                            'previous_id': self.monitor.run.id,
                            'step': self.monitor.run.step,
                            'config': config
                            }, model_path)
                model_path = run_dir / f'{model_name}_{self.save_metric_name}_{best_metric:.4f}_scripted_{task}.pt'
                
                try:
                    scripted_model = torch.jit.script(model)
                    torch.jit.save(scripted_model, model_path,
                               {'config.yaml': yaml.dump(config, encoding='utf-8')})
                except Exception as e:
                    print(f"Error saving scripted model: {e}")

            if self.dry_run:
                break
        self.monitor.finalize()
        
        if self.dry_run:
            print("Deleted wandb folder!")
            shutil.rmtree(Path(self.monitor.run.dir).parent)


def get_metric_name(metric: str, task: str):
    class_name = METRICS[metric].__name__
    
    return f'val_{task.capitalize()}{class_name}'

def get_maximize_metric(metric: str):
    if metric in MINIMZE_METRICS:
        return False
    else:
        return True 