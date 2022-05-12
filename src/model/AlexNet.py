from typing import Tuple, Dict
import torch as t
from torch import nn
from pytorch_lightning import LightningModule

import pandas as pd


class AlexNet(LightningModule):
    def __init__(self,
                 num_classes: int,
                 img_size: int,
                 learning_rate: float,
                 hotel_id_mapping: Dict):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.hotel_id_mapping = hotel_id_mapping

        # https://paperswithcode.com/lib/torchvision/alexnet#
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=3, padding=5),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 192, 5, stride=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(192, 384, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.AvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 4096),  # TODO make dynamic based on img_size
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(),
        )

        self.loss = nn.NLLLoss()

        self.test_df = pd.DataFrame(columns={'image_id', 'hotel_id'})

    def forward(self, x: t.Tensor) -> t.Tensor:
        batch_size, patches, channels, width, height = x.shape

        res = x.reshape(-1, channels, width, height)
        res = self.layers(res)
        res = res.reshape(batch_size, patches, self.num_classes)

        res = self.reduce_patches(res)

        return res

    def reduce_patches(self, x: t.Tensor) -> t.Tensor:
        batch_size, patches, num_classes = x.shape

        log_p = t.sum(x, dim=1)

        norm_log_p = log_p - t.max(log_p, dim=1).values[:,
                             None]  # This is an approximation for normalization could break how loss works. INVESTIGATE maybe.

        return norm_log_p

    def training_step(self, batch: Tuple[t.Tensor, t.Tensor], *args, **kwargs) -> t.Tensor:
        x, y = batch

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        return loss

    def test_step(self, batch: Tuple[t.Tensor, t.Tensor], *args, **kwargs):
        images, image_names = batch

        predictions = self.forward(images)

        for i, prediction in enumerate(predictions):
            # Get top 5 predictions
            top_5 = t.topk(prediction, 5).indices
            hotel_ids = " ".join([str(self.hotel_id_mapping[top_i]) for top_i in top_5.tolist()])

            self.test_df = self.test_df.append({'image_id': image_names[i], 'hotel_id': hotel_ids}, ignore_index=True)

    def on_test_end(self) -> None:
        self.test_df.to_csv('/kaggle/working/submission.csv', columns=['image_id', 'hotel_id'], index=False)

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr=self.learning_rate)