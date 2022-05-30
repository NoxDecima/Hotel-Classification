import os
from typing import Tuple, Dict

import pandas as pd
import torch as t
from tqdm import tqdm


def mean_average_precision(top_k: t.Tensor, targets: t.Tensor, k=5) -> t.Tensor:
    u, _ = predictions.shape

    return t.sum(1 / t.sum((t.triu(t.ones((u, k, k)), diagonal=1) == 0)[top_k == targets[:, None]], dim=1)) / u


def prepare_train_dataframe(path: str) -> Tuple[pd.DataFrame, Dict]:
    hotel_id_mapping = {}

    class_dirs = os.listdir(path)

    df = pd.DataFrame(columns={'class_id', 'image_id', 'path'})
    for i, hotel_id in enumerate(tqdm(class_dirs, desc="Preparing dataset")):
        for dirname, _, images in os.walk(os.path.join(path, hotel_id)):
            for image_id in images:
                df = pd.concat((df, pd.DataFrame({'class_id': [i], 'image_id': [image_id], 'path': [dirname]})),
                               ignore_index=True)

        hotel_id_mapping[i] = int(hotel_id)

    return df, hotel_id_mapping

if __name__ == "__main__":
    df = pd.read_csv("./submission.csv")

    data = prepare_train_dataframe("./data/hotel-id-to-combat-human-trafficking-2022-fgvc9/train_images")

    predictions = t.Tensor([[int(pred) for pred in predictions.split(" ")] for predictions in df['hotel_id'].to_numpy()])


    true_labels = t.Tensor([int(data[1][label]) for label in data[0].sort_values(by='image_id')['class_id'].to_numpy()]).int()

    print(mean_average_precision(predictions, true_labels))




