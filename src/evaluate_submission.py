import pandas as pd
import torch as t

from data.TrainDataModule import prepare_train_dataframe


def mean_average_precision(top_k: t.Tensor, targets: t.Tensor, k=5) -> t.Tensor:
    u, _ = predictions.shape

    return t.sum(1 / t.sum((t.triu(t.ones((u, k, k)), diagonal=1) == 0)[top_k == targets[:, None]], dim=1)) / u

if __name__ == "__main__":
    df = pd.read_csv("./submission.csv")

    data = prepare_train_dataframe("./data/hotel-id-to-combat-human-trafficking-2022-fgvc9/train_images")

    predictions = t.Tensor([[int(pred) for pred in predictions.split(" ")] for predictions in df['hotel_id'].to_numpy()])


    true_labels = t.Tensor([int(data[1][label]) for label in data[0].sort_values(by='image_id')['class_id'].to_numpy()]).int()

    print(mean_average_precision(predictions, true_labels))

