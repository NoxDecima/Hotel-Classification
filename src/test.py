import pytorch_lightning
from torch.utils.data import DataLoader

from model.Transformer import ViT
from data.TestDataset import TestDataset
from data.TrainDataModule import prepare_transforms, TrainDataModule
from data.TrainDataset import TrainDataset

if __name__ == "__main__":
    pytorch_lightning.seed_everything(1234)

    test_images_path = "./data/hotel-id-to-combat-human-trafficking-2022-fgvc9/train_images"
    img_size = 256

    _, val, test_transform = prepare_transforms(img_size)

    #test_dataset = TestDataset(val, test_images_path)

    train_images_path = "./data/small_images"
    val_split = 0.2
    img_size = 256
    batch_size = 64
    shuffle = True

    train_dm = TrainDataModule(
        train_images_path,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        val_split=val_split
    )



    model = ViT.load_from_checkpoint("./logs/lightning_logs/small_ds_s2022/checkpoints/epoch_0009.MAP_1.000000.last.ckpt")

    trainer = pytorch_lightning.Trainer(
        max_epochs=10,
        gpus=1,
    )

    trainer.validate(model, train_dm)



