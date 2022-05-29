import pytorch_lightning
from torch.utils.data import DataLoader

from model.Transformer import ViT
from data.TestDataset import TestDataset
from data.TrainDataModule import prepare_transforms

if __name__ == "__main__":
    pytorch_lightning.seed_everything(1234)

    test_images_path = "./data/hotel-id-to-combat-human-trafficking-2022-fgvc9/test_images"
    img_size = 256

    _, _, test_transform = prepare_transforms(img_size)

    test_dataset = TestDataset(test_transform, test_images_path)

    test_dataloader = DataLoader(test_dataset,
                                  num_workers=8,
                                  batch_size=16,
                                  shuffle=False)

    model = ViT.load_from_checkpoint("./logs/lightning_logs/small_ds_s2022/checkpoints/epoch_0009.MAP_1.000000.last.ckpt")

    trainer = pytorch_lightning.Trainer(
        max_epochs=10,
        gpus=1,
    )

    trainer.test(model, test_dataloader)



