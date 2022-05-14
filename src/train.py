import math

import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from torchvision import transforms
import torch as t

from src.data.TrainDataset import TrainDataset
from src.model.Transformer import ViT


if __name__ == "__main__":

    pytorch_lightning.seed_everything(1234)

    train_images_path = "data/hotel-id-to-combat-human-trafficking-2022-fgvc9/train_images"
    train_masks_path = "data/hotel-id-to-combat-human-trafficking-2022-fgvc9/train_masks"
    val_split = 0.2

    transform = transforms.Compose([transforms.Resize(512),
                                    transforms.CenterCrop(512),
                                    transforms.ToTensor()])

    dataset = TrainDataset(transform,
                           train_images_path,
                           train_masks_path)

    train_set, val_set = t.utils.data.random_split(dataset,
                                                   [int(len(dataset) * (1 - val_split)), int(len(dataset) * val_split)+1])

    train_dataloader = DataLoader(train_set,
                                  num_workers=8,
                                  batch_size=8,
                                  shuffle=True)

    val_dataloader = DataLoader(val_set,
                                num_workers=4,
                                batch_size=16,
                                shuffle=False)

    model = ViT(
        model_kwargs={
            'embed_dim': 256,
            'hidden_dim': 512,
            'num_heads': 8,
            'num_layers': 2,
            'patch_size': 64,
            'num_channels': 3,
            'num_patches': 64,
            'num_classes': 3116,
            'dropout': 0.2
        },
        hotel_id_mapping=dataset.hotel_id_mapping,
        lr=3e-4
    )

    pattern = "epoch_{epoch:04d}.ndcg_{val_ndcg@5_epoch:.6f}"
    ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
    checkpointer = ModelCheckpoint(
        monitor="val_ndcg@5_epoch",
        filename=pattern + ".best",
        mode='max',
        save_last=True,
        auto_insert_metric_name=False,
    )

    trainer = pytorch_lightning.Trainer(
        max_epochs=4,
        gpus=1,
        callbacks=[checkpointer, LearningRateMonitor()],
        default_root_dir="logs",
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
