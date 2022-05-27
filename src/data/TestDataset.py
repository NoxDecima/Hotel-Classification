from typing import Tuple
import torch as t
import pandas as pd
import os

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


class TestDataset(Dataset):
    def __init__(self,
                 transformer,
                 path: str
                 ):
        self.transformer = transformer

        df = pd.DataFrame(columns={'image_id', 'path'})
        for dirname, _, images in tqdm(os.walk(path), desc="Preparing dataset"):
            for image_id in images:
                df = pd.concat((df, pd.DataFrame({'image_id': [image_id], 'path': [dirname]})), ignore_index=True)

        self.df = df

    def __getitem__(self, index) -> Tuple[t.Tensor, t.Tensor]:
        image_path = os.path.join(self.df['path'][index], self.df['image_id'][index])
        img = np.array(Image.open(image_path))

        # If masks are available apply a random mask on image
        if self.transformer is not None:
            img = self.transformer(image=img)['image']

        return img, self.df['image_id'][index]

    def __len__(self):
        return len(self.df)
