import os
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from typing import Type
from src.datamodules.base import DataModule, Stage, Task
from src.datamodules.datasets import JustRAIGSDataset
import torch
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Assumes that the dataset is in the following order:
# JustRAIGS/
#   uncropped/
#     TRAINXXXXXX.JPG
#   cropped/
#     TRAINXXXXXX.JPG
# etc.


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


DATA_PATH = str(Path(__file__).absolute(
).parent.parent.parent / 'data' / 'JustRAIGS')


class JustRAIGSDataModule(DataModule):
    def __init__(self,
                 batch_size: int,
                 test_split: float = 0.1,
                 val_split=0.1,
                 data_dir: str = DATA_PATH,
                 use_cropped: bool = False,
                 task: Task = Task.BINARY,
                 use_nan=False,
                 num_classes=2,
                 num_labels=10,
                 transforms: Type['A.Compose'] = None,
                 positives_only=False,
                 seed=42,
                 negative_resample_step=None,
                 val_samples=None,
                 test_samples=None,
                 dataset=JustRAIGSDataset,
                 num_workers=0,
                 input_size = None,
                 impute = False,
                 balanced = False,
                 k_folds = None,
                 current_fold = None,
                 keras_permute=False):
        super().__init__(task, batch_size, test_split, val_split, seed, num_classes=2,
                         num_labels=10, data_dir=data_dir, transforms=transforms)

        self.k_folds = k_folds
        self.current_fold = current_fold
        
        if self.k_folds is not None and self.current_fold is None:
            raise ValueError("You need to specifiy a current fold for k fold")
        
        self.keras_permute = keras_permute
        self.balanced = balanced
        self.impute = impute
        self.input_size = input_size
        self.num_workers = num_workers
        self.test_samples = test_samples
        self.val_samples = val_samples
        self.negative_resample_step = negative_resample_step

        if negative_resample_step is not None:
            if negative_resample_step < 1:
                raise ValueError('negative_resample_step cannot be lower than 1, otherwise should be None')

        if self.negative_resample_step is not None:
            self.balanced = True
        
        self.positives_only = positives_only
        self.use_nan = use_nan
        if self.task == Task.MULTICLASS.value:
            raise ValueError(
                "Multiclass isn't supported on JustRAIGSDataModule.")

        self.use_cropped = use_cropped

        self.num_classes = 2  # Binary classification (RG vs NRG)

        self.target_transform = v2.Compose([
            v2.ToDtype(torch.float32)
        ])

        self.seed = seed
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)

        if self.transforms is not None:
            equalize = [transform for transform in self.transforms
                          if transform.__class__.__name__ == 'Equalize']
            
            normalize = [transform for transform in self.transforms
                          if transform.__class__.__name__ == 'Normalize'][0]
            
            resize = [transform for transform in self.transforms
                          if transform.__class__.__name__ == 'Resize'][0]
            
            if self.input_size is not None:
                resize = A.Resize(self.input_size, self.input_size, 2)
            

            if len(transforms) >= 2:
                self.to_tensor = A.Compose([*equalize, resize, normalize,
                                            ToTensorV2()])
        else:
            self.to_tensor = A.Compose([ToTensorV2()])

        self.dataset = dataset

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

    def prepare_data(self):
        splits_dir = Path(self.data_dir) / 'splits'

        is_cropped = 'cropped' if self.use_cropped else 'uncropped'
        split_dir = splits_dir / \
            f'{self.seed}_{self.test_split}_{self.val_split}_{is_cropped}'

        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

            dataset = self.dataset(task=Task.BINARY, data_dir=self.data_dir, use_cropped=self.use_cropped)

            indices = dataset.labels_df.index.to_numpy()
            del dataset

            train_indices, self.test_indices = train_test_split(indices,
                                                                test_size=self.test_split,
                                                                random_state=self.seed,
                                                                shuffle=True)

            self.train_indices, self.val_indices = train_test_split(train_indices,
                                                                    test_size=self.val_split,
                                                                    random_state=self.seed,
                                                                    shuffle=True)
            np.savetxt(split_dir / 'train.txt', self.train_indices)
            np.savetxt(split_dir / 'test.txt', self.test_indices)
            np.savetxt(split_dir / 'val.txt', self.val_indices)
        else:
            self.train_indices = np.loadtxt(
                split_dir / 'train.txt', dtype=np.int64)
            self.test_indices = np.loadtxt(
                split_dir / 'test.txt', dtype=np.int64)
            self.val_indices = np.loadtxt(
                split_dir / 'val.txt', dtype=np.int64)

        if self.k_folds is not None:
                dataset = self.dataset(task=Task.BINARY, data_dir=self.data_dir, use_cropped=self.use_cropped)

                indices = dataset.labels_df.index.to_numpy()
                del dataset
                
                self.test_indices = [0]

                kfold = KFold(self.k_folds, shuffle=True, random_state=self.seed)
                
                for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
                    if fold == self.current_fold:
                        self.val_indices = indices[val_ids]
                        self.train_indices = indices[train_ids]
                        print(f'Training with fold {fold}')
                        break

        print(f'Split dataset in train ({len(self.train_indices)}), test ({len(self.test_indices)}) and validation ({len(self.val_indices)})')
        if self.val_samples is not None:
            if self.val_samples > len(self.val_indices):
                raise ValueError(f"Validation samples ({self.val_samples}) larger than validation dataset ({len(self.val_indices)})")

            self.val_indices = self.val_indices[:self.val_samples]
            self.test_indices = self.test_indices[:self.test_samples]
            print(f'Sampling {self.val_samples} indices from validation dataset')

    def setup(self, stage: str):
        if stage == Stage.FIT:
            self.train_dataset = self.dataset(task=self.task,
                                              balanced=self.balanced,
                                              split_indices=self.train_indices,
                                              data_dir=self.data_dir,
                                              use_cropped=self.use_cropped,
                                              transform=self.transforms,
                                              target_transform=self.target_transform,
                                              use_nan=self.use_nan,
                                              positives_only=self.positives_only,
                                              impute=self.impute,
                                              keras_permute=self.keras_permute)
            self.val_dataset = self.dataset(task=self.task,
                                            balanced=False,
                                            split_indices=self.val_indices,
                                            data_dir=self.data_dir,
                                            use_cropped=self.use_cropped,
                                            transform=self.to_tensor,
                                            target_transform=self.target_transform,
                                            use_nan=self.use_nan,
                                            positives_only=self.positives_only,
                                            keras_permute=self.keras_permute)
        elif stage == Stage.TEST:
            self.test_dataset = self.dataset(task=self.task,
                                             balanced=False,
                                             split_indices=self.test_indices,
                                             data_dir=self.data_dir,
                                             use_cropped=self.use_cropped,
                                             transform=self.to_tensor,
                                             target_transform=self.target_transform,
                                             use_nan=self.use_nan,
                                             positives_only=self.positives_only,
                                             keras_permute=self.keras_permute)

        elif stage == Stage.PRED:
            self.pred_dataset = self.dataset(task=self.task,
                                             balanced=False,
                                             split_indices=self.test_indices,
                                             data_dir=self.data_dir,
                                             use_cropped=self.use_cropped,
                                             transform=self.to_tensor,
                                             target_transform=self.target_transform,
                                             use_nan=self.use_nan,
                                             positives_only=self.positives_only,
                                             keras_permute=self.keras_permute)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          worker_init_fn=seed_worker, generator=self.g)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=seed_worker, generator=self.g)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=seed_worker, generator=self.g)

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=seed_worker, generator=self.g)
