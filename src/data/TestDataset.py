from typing import Tuple
import torch as t
import pandas as pd
import os
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
                df = pd.concat(df, {'image_id': image_id, 'path': dirname}, ignore_index=True)

        self.df = df

    def __getitem__(self, index) -> Tuple[t.Tensor, t.Tensor]:
        image_path = os.path.join(self.df['path'][index], self.df['image_id'][index])
        img = Image.open(image_path)

        return self.transformer(img), self.df['image_id'][index]

    def __len__(self):
        return len(self.df)
