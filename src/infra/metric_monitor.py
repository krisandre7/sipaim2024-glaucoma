
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from torchmetrics import Metric, MetricCollection, MetricTracker
import wandb

from src.enums import Task

class MetricMonitor:
    def __init__(self,task: Task, train_metrics: list[Metric], 
                 val_metrics: list[Metric], labels: list[str] = None) -> None:
        self.labels = labels
        
        self.task = task
        if isinstance(self.task, str):
            self.task = Task[self.task.upper()]
        
        # TODO: MetricTracker SEMPRE CONSIDERA O MENOR VALOR MELHOR, MUDAR ISSO
        self.train_metrics = MetricCollection(train_metrics, prefix="train_")
        
        self.val_metrics = MetricCollection(val_metrics, prefix="val_")
        self.train_compute = None
        self.val_compute = None
        
    def to_train(self, device: torch.device):
        self.train_metrics.to(device)

    def to_val(self, device: torch.device):
            self.val_metrics.to(device)
    
    def compute_train(self, plot_conf_matrix=False):
        self.train_compute = self.train_metrics.compute()
        stage_prefix = 'train_'
        
        task_name = self.task.value.title()
        sens_at_spec = f'{stage_prefix}{task_name}SensitivityAtSpecificity'
        if sens_at_spec in self.train_compute:
            self.train_compute['train_threshold'] = self.train_compute[sens_at_spec][1]
            self.train_compute[sens_at_spec] = self.train_compute[sens_at_spec][0]
        
        class_name = f'{task_name}ConfusionMatrix'
        if plot_conf_matrix and hasattr(self.train_metrics, class_name):
            conf_matrix = getattr(self.train_metrics, class_name)
            key = f'{stage_prefix}{class_name}'
            
            fig, ax = conf_matrix.plot(self.train_compute[key], 
                                       labels=self.labels)
            self.train_compute[key] = fig
            plt.close()
        
        return self.train_compute

    def compute_val(self, plot_conf_matrix: bool = False):
        self.val_compute = self.val_metrics.compute()
        stage_prefix = 'val_'
        
        task_name = self.task.value.title()
        sens_at_spec = f'{stage_prefix}{task_name}SensitivityAtSpecificity'
        if sens_at_spec in self.val_compute:
            self.val_compute['val_threshold'] = self.val_compute[sens_at_spec][1]
            self.val_compute[sens_at_spec] = self.val_compute[sens_at_spec][0]
        
        class_name = f'{task_name}ConfusionMatrix'
        if plot_conf_matrix and hasattr(self.val_metrics, class_name):
            conf_matrix = getattr(self.val_metrics, class_name)
            key = f'{stage_prefix}{class_name}'
            
            fig, ax = conf_matrix.plot(self.val_compute[key], 
                                       labels=self.labels)
            self.val_compute[key] = wandb.Image(fig)
            plt.close()
        
        return self.val_compute
    
    def update_train(self, logits, y_true):
        if logits.dim() < y_true.dim():
            logits = logits.unsqueeze(-1)
        else:    
            logits = logits.squeeze()

        preds = torch.sigmoid(logits)
        try:
            self.train_metrics.update(preds, y_true)
        except RuntimeError:
            self.train_metrics.update(preds, y_true.int())

    def update_val(self, logits, y_true):
        if logits.dim() < y_true.dim():
            logits = logits.unsqueeze(-1)
        else:    
            logits = logits.squeeze()

        preds = torch.sigmoid(logits)
        try:
            self.val_metrics.update(preds, y_true)
        except RuntimeError:
            self.val_metrics.update(preds, y_true.int())
    
    def reset(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
        
    def train_metrics_str(self):
        
        string = ''
        for metric_name, value in self.train_compute.items():
            if isinstance(value, tuple):
                value = value[0]
                
            if value.dim() > 1:
                continue
            string += f', {metric_name}: '

            if value.dim() == 1:
                # labels = self.labels
                # if labels is None:
                #     labels = torch.arange(0,len(value))
                string += '['
                for idx, val in enumerate(value):
                    string += f'{val:.2f}, '
                string += ']'
            else:
                string += f'{value:.2f}'
        return string

    def val_metrics_str(self):
        
        string = ''
        for metric_name, value in self.val_compute.items():
            if isinstance(value, tuple):
                value = value[0]
                
            if value.dim() > 1:
                continue
            string += f', {metric_name}: '

            if value.dim() == 1:
                # labels = self.labels
                # if labels is None:
                #     labels = torch.arange(0,len(value))
                string += '['
                for idx, val in enumerate(value):
                    string += f'{val:.2f}, '
                string += ']'
            else:
                string += f'{value:.2f}'
        return string