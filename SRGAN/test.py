from model import Generator, Discriminator
import torch
from dataloader import get_loader
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import cv2
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen_model = Generator().to(device)

gen_opt = torch.optim.Adam(gen_model.parameters(), lr=1e-4, betas=(0.9, 0.999))

lr = 1e-4

load_checkpoint("gen.pth.tar", gen_model, gen_opt, lr)
valid_loader = get_loader("data/HR/DIV2K_valid_HR", 16, True)
gen_model.eval()

for high_res, low_res in valid_loader:
    hr_image = high_res.to(device)
    lr_image = low_res.to(device)
    with torch.no_grad():
        u_img = gen_model(lr_image)
        u_img = u_img.cpu()
        hr_image = hr_image.cpu()
        lr_image = lr_image.cpu()
        count = 9
        count2 = 9
        count3 = 9
        for i in range(1, 10):
            plt.subplot(3, 3, i)
            if i % 3 == 1:
                save_image(u_img[count], "gen_img.png")
                count = count + 1

            elif i % 3 == 2:
                save_image(lr_image[count2], "low_res.png")
                count2 = count2 + 1
            else:
                save_image(hr_image[count3], "original.png")
                count3 = count3 + 1
        break
#
#image = Image.open('low_res.png')
#
#transform = transforms.Compose([
# transforms.Resize((64, 64)),
# transforms.ToTensor(),
# transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
#])
#
#lr_image = transform(image)
#
#with torch.no_grad():
# lr_image = lr_image.to(device)
# lr_image = lr_image.unsqueeze(0)
# sr_image = gen_model(lr_image)
# sr_image = sr_image.cpu()
# save_image(sr_image, "gen_img.png")
