import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from model.Transformer import ViT
from data.TrainDataModule import TrainDataModule

if __name__ == "__main__":
    pytorch_lightning.seed_everything(1234)

    train_images_path = "./data/hotel-id-to-combat-human-trafficking-2022-fgvc9/train_images"
    val_split = 0.2
    img_size = 512
    batch_size = 16
    shuffle = True

    train_dm = TrainDataModule(
        train_images_path,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        val_split=val_split
    )

    model = ViT(
        model_kwargs={
            'embed_dim': 256,
            'hidden_dim': 512,
            'num_heads': 16,
            'num_layers': 6,
            'patch_size': 64,
            'num_channels': 3,
            'num_patches': 64,
            'num_classes': 3116,
            'dropout': 0.5
        },
        hotel_id_mapping=train_dm.hotel_id_mapping,
        lr=3e-4
    )

    pattern = "epoch_{epoch:04d}.MAP_{val_MAP@5:.6f}"
    ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
    checkpointer = ModelCheckpoint(
        monitor="val_MAP@5",
        filename=pattern + ".best",
        mode='max',
        save_last=True,
        auto_insert_metric_name=False,
    )

    trainer = pytorch_lightning.Trainer(
        max_epochs=30,
        gpus=1,
        callbacks=[checkpointer, LearningRateMonitor()],
        default_root_dir="logs",
    )

    trainer.fit(model, train_dm)
