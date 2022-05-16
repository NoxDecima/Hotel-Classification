from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
from torch import optim, nn
import torch as t

from src.module.VisionTransformer import VisionTransformer

"""
Code copied from 
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
"""


class ViT(pl.LightningModule):

    def __init__(self, model_kwargs: dict, hotel_id_mapping: dict, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)

        self.hotel_id_mapping = hotel_id_mapping

        self.loss = nn.CrossEntropyLoss()
        self.num_classes = model_kwargs['num_classes']

        self.test_df = pd.DataFrame(columns={'image_id', 'hotel_id'})

    def forward(self, x):
        return self.model(x)


    def mean_average_precision(self, predictions: t.Tensor, targets: t.Tensor, k=5) -> t.Tensor:
        u, _ = predictions.shape

        top_k = t.topk(predictions, k, dim=1).indices

        scores = t.zeros(targets.shape)

        for i in range(k):
            relevance = t.zeros((u, i+1))
            relevance[:, i] = 1

            scores += t.mean((top_k[:, :i+1] == targets[:, None]).float() * relevance, dim=1)

        return t.mean(scores)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch: Tuple[t.Tensor, t.Tensor], mode: str = "train") -> t.Tensor:
        x, y = batch

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        acc = (y_hat.argmax(dim=-1) == y).float().mean()

        self.log(f"{mode}_MAP@5", self.mean_average_precision(y_hat, y, k=5),
                 prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{mode}_loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True)

        self.log(f"{mode}_acc", acc,
                 prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch: Tuple[t.Tensor, t.Tensor], *args, **kwargs) -> t.Tensor:
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch: Tuple[t.Tensor, t.Tensor], *args, **kwargs):
        self._calculate_loss(batch, mode="val")

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