from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CyclicLR, CosineAnnealingLR
from src.datamodules.base import DataModule
from src.datamodules.justraigs import JustRAIGSDataModule

from src.models import (ReferralModel, JustificationModel, JustExtractionModel)
from src.datamodules.datasets import JustRAIGSDataset
from torch.utils.data import Dataset
from torchmetrics import Accuracy, ConfusionMatrix, HammingDistance, F1Score, Specificity, Recall, Precision

from src.metrics import SensitivityAtSpecificity

from typing import Dict


MODELS: Dict[str, nn.Module]  = {
    "referral": ReferralModel,
    "justification": JustificationModel,
    'just_extraction': JustExtractionModel
}

DATASETS: Dict[str, Dataset] = {
    "justraigs": JustRAIGSDataset
}

DATA: Dict[str, Dataset] = {
    "justraigs": JustRAIGSDataset
}

# TRAINERS = {
#     # Default Trainer
#     None: Trainer
# }

DATAMODULES: dict[str, DataModule] = {
    'justraigs': JustRAIGSDataModule
}

LOSSES: Dict[str, nn.Module] = {
    "bce_loss": nn.BCELoss,
    "mse_loss": nn.MSELoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "bce_logits_loss": nn.BCEWithLogitsLoss
}

OPTIMIZERS: Dict[str, nn.Module]  = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
}

SCHEDULERS = {
    "reduce_on_plateau": ReduceLROnPlateau,
    "exponential": ExponentialLR,
    "cyclic": CyclicLR,
    "cosine": CosineAnnealingLR
}

METRICS = {
    "accuracy": Accuracy,
    "conf_matrix": ConfusionMatrix,
    "ham_dist": HammingDistance,
    "sens_at_spec": SensitivityAtSpecificity,
    "f1_score": F1Score,
    "specificity": Specificity,
    "recall": Recall,
    "precision": Precision
}

# WHEN ADDING A METRIC DON'T FORGET TO SPECIFY HERE IF IT'S SUPPOSED TO BE MINIMIZED
MINIMZE_METRICS = {'ham_dist', 'val_loss'}