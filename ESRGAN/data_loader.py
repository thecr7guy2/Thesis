import cv2
import improc
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt
import numpy as np


class SuperRes(Dataset):
    def __init__(self, hr_data_dir, hr_image_size, upscale_factor, mode):
        super(SuperRes, self).__init__()
        self.hr_data_dir = hr_data_dir
        self.image_list = os.listdir(hr_data_dir)
        self.image_list = sorted(self.image_list)
        self.hr_image_size = hr_image_size
        self.upscale_factor = upscale_factor
        self.mode = mode

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_name = self.image_list[index]
        image = cv2.imread(os.path.join(self.hr_data_dir, img_name)).astype(np.float32) / 255.
        if self.mode == "Train":
            hr_image = improc.random_crop(image, self.hr_image_size)
            hr_image = improc.random_rotate(hr_image, [90, 180, 270])
            hr_image = improc.random_horizontally_flip(hr_image, 0.5)
            hr_image = improc.random_vertically_flip(hr_image, 0.5)
        elif self.mode == "Valid":
            hr_image = improc.center_crop(image, self.hr_image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")

        lr_image = improc.image_resize(hr_image, 1 / self.upscale_factor)

        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        hr_tensor = improc.image_to_tensor(hr_image, False, False)
        lr_tensor = improc.image_to_tensor(lr_image, False, False)

        return hr_tensor, lr_tensor


def get_loader(hr_data_dir, hr_image_size, upscale_factor, mode, batch_size, shuffle):
    dataset = SuperRes(hr_data_dir, hr_image_size, upscale_factor, mode)
    d_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=3)
    return d_loader


#
# data_loader = get_loader("../SRGAN/data/HR/DIV2K_valid_HR", 256, 4, "Train", 16, True)
# a, b = next(iter(data_loader))
#
# plt.imshow(np.transpose(utils.make_grid(a[:16], padding=2, normalize=True), (1, 2, 0)))
# plt.show()
# plt.imshow(np.transpose(utils.make_grid(b[:16], padding=2, normalize=True), (1, 2, 0)))
# plt.show()
