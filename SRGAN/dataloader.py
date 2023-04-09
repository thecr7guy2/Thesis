from PIL import Image
from PIL.Image import Resampling
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Dataset
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np

#######################
hr_size = 96
upscale_factor = 4
######################
primary_transform = A.Compose(
    [
        A.RandomCrop(width=hr_size, height=hr_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

hr_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ]
)
lr_transform = A.Compose(
    [
        A.Resize(width=hr_size // upscale_factor, height=hr_size // upscale_factor, interpolation=Resampling.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)


class superres(Dataset):
    def __init__(self, hr_data_dir, hr_transform, lr_transform, primary_transform):
        self.hr_data_dir = hr_data_dir
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        self.primary_transform = primary_transform
        self.image_list = os.listdir(hr_data_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_name = self.image_list[index]
        image = np.array(Image.open(os.path.join(self.hr_data_dir, img_name)))
        image = self.primary_transform(image=image)["image"]
        hr_image = self.hr_transform(image=image)["image"]
        lr_image = self.lr_transform(image=image)["image"]
        return hr_image, lr_image


def get_loader(hr_data_dir, batch_size, shuffle):
    dataset = superres(hr_data_dir, hr_transform, lr_transform, primary_transform)
    d_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=3)
    return d_loader


# data_loader = get_loader("data/HR/DIV2K_train_HR", 16, True)
# for high_res, low_res, in data_loader:
#     print(low_res.shape)
#     print(high_res.shape)
#     break
# a, b = next(iter(data_loader))
#
# plt.imshow(np.transpose(utils.make_grid(a[:16], padding=2, normalize=True), (1, 2, 0)))
# plt.show()
# plt.imshow(np.transpose(utils.make_grid(b[:16], padding=2, normalize=True), (1, 2, 0)))
# plt.show()
