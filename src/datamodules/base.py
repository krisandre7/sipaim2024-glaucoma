from pathlib import Path
from typing import Optional, Type
from src import Task, Stage

from torch.utils.data import DataLoader


class DataModule:

    def __init__(self, task: Task, batch_size: int, test_split: float, val_split: float, seed: int = 42,
                 num_classes: Optional[int] = 0, num_labels: Optional[int] = 0,
                 data_dir: str = None, transforms: Type['Compose'] = None):
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pred_dataset = None
        self.test_split = test_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transforms = transforms
        self.seed = seed

    def prepare_data(self) -> None:
        """
        Implement to perform any data preparation steps such as downloading or preprocessing the dataset.
        """
        raise NotImplementedError("Please implement prepare_data method in your subclass.")

    def setup(self, stage: Stage) -> None:
        """
        Implement to set up the dataset for training, validation, and testing stages.

        Args:
            stage (str): One of 'fit', 'validate', or 'test'. It indicates which stage of setup is being called.
        """
        raise NotImplementedError("Please implement setup method in your subclass.")

    def train_dataloader(self) -> DataLoader:
        """
        Implement to return the data loader for training dataset.

        Returns:
            torch.utils.data.DataLoader: The data loader for training dataset.
        """
        raise NotImplementedError("Please implement train_dataloader method in your subclass.")

    def val_dataloader(self) -> DataLoader:
        """
        Implement to return the data loader for validation dataset.

        Returns:
            torch.utils.data.DataLoader: The data loader for validation dataset.
        """
        raise NotImplementedError("Please implement val_dataloader method in your subclass.")

    def test_dataloader(self) -> DataLoader:
        """
        Implement to return the data loader for test dataset.

        Returns:
            torch.utils.data.DataLoader: The data loader for test dataset.
        """
        raise NotImplementedError("Please implement test_dataloader method in your subclass.")

    def predict_dataloader(self) -> DataLoader:
        """
        Implement to return the data loader for prediction dataset.

        Returns:
            torch.utils.data.DataLoader: The data loader for prediction dataset.
        """
        raise NotImplementedError("Please implement predict_dataloader method in your subclass.")