from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt
import numpy as np


class Animefaces(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_name = self.image_list[index]
        img = Image.open(os.path.join(self.data_dir, img_name))

        if self.transform is not None:
            img = self.transform(img)

        return img


def get_loader(data_dir, transform, batch_size, shuffle):
    anime_dataset = Animefaces(data_dir, transform)
    anime_loader = DataLoader(anime_dataset, batch_size=batch_size, shuffle=shuffle)
    return anime_loader


# anime_transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])
# data_loader = get_loader("data/images", anime_transform, 16, shuffle=False)
# images = next(iter(data_loader))
# rows = 4
# columns = 4
# print(images.shape)
# print(images[0].max())
# # fig, ax = plt.subplots(4, 4, figsize=(30, 20))
# # ax = ax.ravel()
# # for i in range(0, 16):
# #     ax[i].imshow(images[i].numpy().transpose((1, 2, 0)))
# # plt.show()
