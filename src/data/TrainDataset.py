from torchvision import transforms
from typing import Tuple
import torch as t
import random
import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class TrainDataset(Dataset):
    def __init__(self,
                 transformer,
                 path: str,
                 mask_path: str = None
                 ):
        # Images [batch, segments, colour, height, width]

        self.transformer = transformer
        self.mask_path = mask_path
        self.hotel_id_mapping = {}

        class_dirs = os.listdir(path)

        df = pd.DataFrame(columns={'hotel_id', 'image_id', 'path'})
        for i, hotel_id in enumerate(tqdm(class_dirs, desc="Preparing dataset")):
            for dirname, _, images in os.walk(os.path.join(path, hotel_id)):
                for image_id in images:
                    df = pd.concat([df, pd.DataFrame({'hotel_id': [hotel_id], 'class_id': [i], 'image_id': [image_id], 'path': [dirname]})],ignore_index=True)
            self.hotel_id_mapping[i] = hotel_id

        # TODO save dataframe and load it for known dataset.

        self.df = df

    def __getitem__(self, index) -> Tuple[t.Tensor, t.Tensor]:
        image_path = os.path.join(self.df['path'][index], self.df['image_id'][index])
        img = Image.open(image_path)

        # If masks are avaliable apply a random mask on image
        if self.mask_path != None:
            mask = Image.open(os.path.join(self.mask_path, random.choice(
                [m for m in os.listdir(self.mask_path) if os.path.isfile(os.path.join(self.mask_path, m))])))
            # For some reason the size of the mask is flipped with the size of the image
            mask_transform = transforms.Compose([transforms.Resize(img.size)])
            tf_mask = mask_transform(mask)
            img.paste(tf_mask, (0, 0), tf_mask)

        return self.transformer(img), int(self.df['class_id'][index])

    def __len__(self):
        return len(self.df)
