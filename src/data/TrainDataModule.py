from typing import Tuple, Any, Dict
import os

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import pandas as pd
import numpy as np

import cv2
import albumentations as a
import albumentations.pytorch as apt
from torch.utils.data import DataLoader

from data.TrainDataset import TrainDataset


class TrainDataModule(LightningDataModule):
    def __init__(self,
                 path: str,
                 img_size: int,
                 batch_size: int,
                 shuffle: bool,
                 val_split: float = 0.2):
        data_df, self.hotel_id_mapping = prepare_train_dataframe(path)

        train_df, val_df = np.split(data_df.sample(frac=1), [int((1 - val_split) * len(data_df))])

        train_transform, val_transform, _ = prepare_transforms(img_size)

        image_path = os.path.join(path, "images")

        train_set = TrainDataset(image_path, train_df.reset_index(), train_transform)
        val_set = TrainDataset(image_path, val_df.reset_index(), val_transform)

        self.train_data = DataLoader(train_set,
                                      num_workers=8,
                                      batch_size=batch_size,
                                      shuffle=shuffle)

        self.val_data = DataLoader(val_set,
                                    num_workers=8,
                                    batch_size=batch_size,
                                    shuffle=False)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_data

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_data


def prepare_train_dataframe(path: str) -> Tuple[pd.DataFrame, Dict]:
    data_df = pd.read_csv(os.path.join(path, "train.csv"))
    # encode hotel ids
    data_df["hotel_id_code"] = data_df["hotel_id"].astype('category').cat.codes.values.astype(np.int64)
    # convert hotel_id encoding to dict
    hotel_id_code_df = data_df.drop(columns=["image_id"]).drop_duplicates().reset_index(drop=True)
    hotel_id_mapping = hotel_id_code_df.set_index('hotel_id_code').to_dict()["hotel_id"]

    return data_df, hotel_id_mapping


def prepare_transforms(img_size: int) -> Tuple[Any, Any, Any]:
    train_transform = a.Compose([
        a.Resize(img_size, img_size),
        a.HorizontalFlip(p=0.75),
        a.VerticalFlip(p=0.25),
        a.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT),
        a.OpticalDistortion(p=0.25),
        a.Perspective(p=0.25),
        a.CoarseDropout(p=0.5, min_holes=1, max_holes=6,
                        min_height=img_size // 16, max_height=img_size // 4,
                        min_width=img_size // 16, max_width=img_size // 4),  # normal coarse dropout

        a.CoarseDropout(p=1., max_holes=1,
                        min_height=img_size // 4, max_height=img_size // 2,
                        min_width=img_size // 4, max_width=img_size // 2,
                        fill_value=(255, 0, 0)),  # simulating occlusions in test data

        a.RandomBrightnessContrast(p=0.75),
        a.ToFloat(),
        apt.transforms.ToTensorV2(),
    ])

    # used for validation dataset - only occlusions
    val_transform = a.Compose([
        a.Resize(img_size, img_size),
        a.CoarseDropout(p=1., max_holes=1,
                        min_height=img_size // 4, max_height=img_size // 2,
                        min_width=img_size // 4, max_width=img_size // 2,
                        fill_value=(255, 0, 0)),  # simulating occlusions
        a.ToFloat(),
        apt.transforms.ToTensorV2(),
    ])

    # no augmentations
    test_transform = a.Compose([
        # TODO add padding !!!
        a.Resize(img_size, img_size),
        a.ToFloat(),
        apt.transforms.ToTensorV2(),
    ])

    return train_transform, val_transform, test_transform
