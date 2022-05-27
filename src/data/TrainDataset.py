import numpy as np
from pandas import DataFrame
from typing import Tuple
import torch as t
import os
from torch.utils.data import Dataset
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self,
                 data: DataFrame,
                 transformer=None,
                 ):
        # Images [batch, segments, colour, height, width]
        self.transformer = transformer
        self.df = data

    def __getitem__(self, index) -> Tuple[t.Tensor, int]:
        image_path = os.path.join(self.df['path'][index], self.df['image_id'][index])
        img = np.array(Image.open(image_path))

        # If masks are available apply a random mask on image
        if self.transformer is not None:
            img = self.transformer(image=img)['image']

        return img, self.df['class_id'][index]

    def __len__(self):
        return len(self.df)
