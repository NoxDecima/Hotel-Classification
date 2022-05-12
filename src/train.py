import math

import pytorch_lightning
from torch.utils.data import DataLoader
from torchvision import transforms
import torch as t

from src.data.TrainDataset import TrainDataset
from src.model.AlexNet import AlexNet


def generatePatches(img: t.Tensor):
    nr_of_patches_per_axis = 4
    patch_size_w = math.floor(img.shape[1] / nr_of_patches_per_axis)
    patch_size_h = math.floor(img.shape[2] / nr_of_patches_per_axis)

    patches = img.unfold(0, 3, 3).unfold(1, patch_size_w, patch_size_h).unfold(2, patch_size_w, patch_size_h)

    patches = t.reshape(patches, (nr_of_patches_per_axis ** 2, 3, patch_size_w, patch_size_h))

    return patches


if __name__ == "__main__":

    train_images_path = "/kaggle/input/hotel-id-to-combat-human-trafficking-2022-fgvc9/train_images"
    train_masks_path = "/kaggle/input/hotel-id-to-combat-human-trafficking-2022-fgvc9/train_masks"

    test_images_path = ""

    transform = transforms.Compose([transforms.Resize(1024),
                                    transforms.CenterCrop(1024),
                                    transforms.ToTensor(),
                                    transforms.Lambda(generatePatches)])

    train_dataset = TrainDataset(transform,
                                 train_images_path,
                                 train_masks_path)

    train_dataloader = DataLoader(train_dataset,
                                  num_workers=2,
                                  batch_size=16,
                                  shuffle=True)

    model = AlexNet(3116,
                    244,
                    0.001,
                    train_dataset.hotel_id_mapping)

    trainer = pytorch_lightning.Trainer(
        max_epochs=1,
        gpus=1
    )

    trainer.fit(model, train_dataloader)
