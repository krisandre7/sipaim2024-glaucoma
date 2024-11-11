from enum import Enum
import glob
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
from src.enums import Task
from src.utils import compare_images
from torchvision.transforms import v2
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Assumes that the dataset is in the following order:
# JustRAIGS/
#   uncropped/
#     TRAINXXXXXX.JPG
#   cropped/
#     TRAINXXXXXX.JPG
# etc.

# The following Eye IDs are present only in the label dataset, but not in the image files
NOT_IN_FILES = ['TRAIN000426', 'TRAIN000995', 'TRAIN004725', 'TRAIN005747', 'TRAIN007578',
                'TRAIN008333', 'TRAIN013811', 'TRAIN014086', 'TRAIN015491', 'TRAIN017381',
                'TRAIN018451', 'TRAIN018824', 'TRAIN019811', 'TRAIN020318', 'TRAIN023419',
                'TRAIN024724', 'TRAIN026781', 'TRAIN030735', 'TRAIN032842', 'TRAIN034377',
                'TRAIN035808', 'TRAIN041477', 'TRAIN042478', 'TRAIN043432', 'TRAIN043903',
                'TRAIN044082', 'TRAIN044224', 'TRAIN044421', 'TRAIN049684', 'TRAIN050491',
                'TRAIN052938', 'TRAIN053302', 'TRAIN055868', 'TRAIN059062', 'TRAIN063225',
                'TRAIN064444', 'TRAIN066487', 'TRAIN067271', 'TRAIN068193', 'TRAIN068882',
                'TRAIN070830', 'TRAIN072002', 'TRAIN072786', 'TRAIN073835', 'TRAIN074494',
                'TRAIN078960', 'TRAIN079580', 'TRAIN079727', 'TRAIN080190', 'TRAIN088906',
                'TRAIN094983', 'TRAIN097004', 'TRAIN098628', 'TRAIN100738', 'TRAIN101135']

DATA_PATH = Path(__file__).absolute().parent.parent.parent.parent / 'data' / 'JustRAIGS'

LABELS_FILE = "JustRAIGS_Train_labels.csv"
FEATURE_COLUMNS = ['ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']


class JustRAIGSDataset(Dataset):
    def __init__(self, task: Task, split_indices=None, data_dir: str = None,
                 transform: v2.Transform = None, target_transform: v2.Transform = None,
                 use_cropped: bool = False, balanced=False,
                 positives_only=False, negatives_only=False,
                 use_nan=False, impute=False, keras_permute=False, **kwargs):

        self.keras_permute = keras_permute
        self.impute = impute
        self.task = task
        self.balanced = balanced

        self.positives_only = positives_only
        self.negatives_only = negatives_only

        if self.positives_only and self.negatives_only:
            raise ValueError("positives_only and negatives_only cannot be true at the same time.")

        if self.positives_only or self.negatives_only:
            self.balanced = False

        if isinstance(self.task, str):
            self.task = Task[self.task.upper()]

        if self.task == Task.MULTICLASS:
            return ValueError("This class does not support multiclass")

        self.use_nan = use_nan
        self.use_cropped = use_cropped

        self.data_dir = Path(data_dir) if data_dir is not None else DATA_PATH
        self.transform = transform
        self.target_transform = target_transform

        self.labels_df = pd.read_csv(self.data_dir / LABELS_FILE, sep=";")

        self.labels_df = self.labels_df[~self.labels_df['Eye ID'].isin(NOT_IN_FILES)]
        self.labels_df.reset_index(drop=True, inplace=True)

        # Filter labels of cropped or uncropped images
        image_dir = 'cropped' if self.use_cropped else 'uncropped'

        image_paths = glob.glob(f"{self.data_dir / image_dir}/*.JPG")

        # Extract Eye ID from images TRAINXXXXXX_cropped.JPG
        image_ids = set(map(lambda x: os.path.basename(x)[:11], image_paths))

        self.labels_df = self.labels_df[self.labels_df['Eye ID'].isin(image_ids)]
        self.labels_df.reset_index(drop=True, inplace=True)

        if split_indices is not None:
            self.labels_df = self.labels_df.iloc[split_indices]
            self.labels_df.reset_index(drop=True, inplace=True)

        if self.positives_only:
            self.labels_df = self.labels_df[self.labels_df['Final Label'] == 'RG']
            self.labels_df.reset_index(drop=True, inplace=True)
        elif self.negatives_only:
            self.labels_df = self.labels_df[self.labels_df['Final Label'] == 'NRG']
            self.labels_df.reset_index(drop=True, inplace=True)

        if self.balanced:
            self.original_labels_df = self.labels_df.copy()
            self.labels_df = undersample_df(self.labels_df)

        self.feature_labels_df = self.get_feature_labels(self.labels_df)
        
        if self.impute:
            imp = IterativeImputer(max_iter=10000, random_state=42)
            imp.fit(self.feature_labels_df.iloc[:, :-1])
            self.feature_labels_df.iloc[:, :-1] = np.round(imp.transform(self.feature_labels_df.iloc[:, :-1]))
            

        self.label_encoder = {"RG": 1, "NRG": 0}

        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])

    def __getitem__(self, index):
        # Preciso fazer isso para garantir que o dataset funcione 🥴
        if self.task == Task.BINARY:
            sample = self.labels_df.iloc[index]
        else:
            sample = self.feature_labels_df.iloc[index]

        image_path = self.get_image_path(index)

        image = Image.open(image_path)

        if self.transform:
            image = np.array(image)
            image = self.transform(image=image)['image'].float()
        else:
            image = np.array(image)
            image = self.to_tensor(image)
        
        if self.keras_permute:
            image = image.permute(1, 2, 0)
        

        if self.task == Task.BINARY:
            label = self.labels_df.loc[index, "Final Label"]
            label = self.label_encoder[label]
        else:
            label = self.feature_labels_df.iloc[index, :-1].astype('Int64').to_numpy()
            label = torch.tensor(label, dtype=torch.float32)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.labels_df) if self.task is Task.BINARY else len(self.feature_labels_df)

    def resample_negatives(self):
        num_neg_samples = len(self.labels_df[self.labels_df['Final Label'] == 'NRG'])

        new_neg_samples = self.original_labels_df[
            self.original_labels_df['Final Label'] == 'NRG'].sample(num_neg_samples)

        self.labels_df.loc[self.labels_df['Final Label'] == 'NRG'] = new_neg_samples.to_numpy()
        self.feature_labels_df = self.get_feature_labels(self.labels_df)

    def get_image_path(self, index: int):
        image_id = self.labels_df.loc[index, "Eye ID"]
        image_name = f"{image_id}_cropped.JPG" if self.use_cropped else f"{image_id}.JPG"

        if self.use_cropped:
            return self.data_dir / "cropped" / image_name
        else:
            return self.data_dir / 'uncropped' / image_name

    def get_image_id(self, index: int):
        image_id = self.labels_df.loc[index, "Eye ID"]
        return image_id

    def get_feature_labels(self, labels_df: pd.DataFrame):

        # Dataframe with the final labels of the 10 features
        feature_labels_df = pd.DataFrame(index=np.arange(len(labels_df)), columns=FEATURE_COLUMNS)

        features_df = labels_df.iloc[:, 7:]
        g1_features = features_df.iloc[:, 10 * 0:10 * 1]
        g2_features = features_df.iloc[:, 10 * 1:10 * 2]
        g3_features = features_df.iloc[:, 10 * 2:10 * 3]

        # To compare dataframes they need to have the same shape and column names.
        g1_features.columns = g2_features.columns
        agree_mask = g1_features == g2_features

        # Add all rows in which G1 and G2 agree
        agree_rows = agree_mask.all(axis=1)
        feature_labels_df[agree_rows] = g1_features[agree_rows]

        # Add all rows in which G1 and G2 disagree but G3 is not Nan
        nan_rows = g3_features.isna().all(axis=1).to_numpy()
        g3_not_nan = np.logical_and(~nan_rows, ~agree_rows)
        feature_labels_df[g3_not_nan] = g3_features[g3_not_nan]

        # Get all rows in which G1 and G2 disagree but G3 is Nan
        g3_nan = np.logical_and(~agree_rows, nan_rows)

        # Get all rows where at least one column is false (disagreement) and G3 is Nan
        # Fill the False columns with nan
        # Fill the True columns with the agreed values

        at_least_one = ~agree_rows.to_numpy()
        at_least_one_nan = np.logical_and(at_least_one, g3_nan)

        if self.use_nan:
            feature_labels_df[at_least_one_nan] = g2_features[agree_mask][at_least_one_nan].to_numpy()
        else:
            feature_labels_df.drop(index=np.where(at_least_one_nan)[0], inplace=True)

        feature_labels_df = feature_labels_df.astype(np.float32)
        feature_labels_df.loc[:, 'Eye ID'] = labels_df.loc[feature_labels_df.index, 'Eye ID']
        feature_labels_df.reset_index(drop=True, inplace=True)

        return feature_labels_df


def undersample_df(df: pd.DataFrame, target_column='Final Label'):
    class_counts = df[target_column].value_counts()

    # Find majority and minority class based on counts
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    # Get the number of samples for the minority class
    minority_count = df[df[target_column] == minority_class].shape[0]

    # Undersample the majority class to match the size of the minority class
    majority_df = df[df[target_column] == majority_class]
    undersampled_majority = majority_df.sample(minority_count, replace=False)

    # Combine the undersampled majority class with the minority class
    undersampled_df = pd.concat([undersampled_majority, df[df[target_column] == minority_class]], ignore_index=True)

    return undersampled_df


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    from torchvision.transforms import v2

    dataset = JustRAIGSDataset()
    index = random.randint(0, len(dataset) - 1)

    image, label = dataset[index]
    image_name = dataset.get_image_name(0)

    print(f"Image {image_name} label is: {label}, shape is {np.asarray(image).shape}")
    # plt.imshow(image)
    # plt.show()
