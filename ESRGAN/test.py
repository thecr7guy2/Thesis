from model import RRDBNet
import torch
from data_loader import get_loader
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch import optim
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import cv2
from PIL import Image
from PIL.Image import Resampling
import torchvision.transforms as transforms
from patchify import patchify, unpatchify
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_checkpoint(checkpoint_file, model, optimizer, scheduler):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    curr_epoch = checkpoint["epoch"]
    ssim = checkpoint["ssim"]
    psnr = checkpoint["psnr"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    lr_epoch = scheduler.get_last_lr()
    lr_epoch = lr_epoch[0]
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_epoch
    return curr_epoch, psnr, ssim


epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = RRDBNet(3, 3, 64, 32, 2, 0.2).to(device)

gen_opt = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)

model_mile = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]

gen_scheduler = torch.optim.lr_scheduler.MultiStepLR(gen_opt, model_mile, 0.5)

gen2 = RRDBNet(3, 3, 64, 32, 2, 0.2).to(device)

gen_opt2 = torch.optim.Adam(gen2.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)

model_mile = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]

gen_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(gen_opt2, model_mile, 0.5)

a, b, c = load_checkpoint("gen25.pth.tar", gen, gen_opt, gen_scheduler)
d, e, f = load_checkpoint("gen50.pth.tar", gen2, gen_opt2, gen_scheduler2)
valid_loader = get_loader("../SRGAN/data/HR/DIV2K_valid_HR", 1, True)
gen.eval()


# high_res, low_res, c = next(iter(valid_loader))
# hr_image = high_res.to(device)
# lr_image = low_res.to(device)
# with torch.no_grad():
#     u_img = gen(lr_image)
#     u_img2 = gen2(lr_image)
#     u_img = u_img.cpu()
#     u_img2 = u_img2.cpu()
#     hr_image = hr_image.cpu()
#     lr_image = lr_image.cpu()
#     print(c)
#     save_image(u_img, "gen_img.png")
#     save_image(u_img2, "gen_img2.png")
#     save_image(lr_image, "low_res.png")
#     save_image(hr_image, "original.png")

image = Image.open('target.jpg')
height, width = image.size
print(width, height)
transform1 = transforms.Compose([
    transforms.RandomCrop((256, 256)),
])
transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
])

transform3 = transforms.Compose([
    transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
])

ghj = transform1(image)
lr_image = transform3(ghj)
hr_image = transform2(ghj)

with torch.no_grad():
    lr_image = lr_image.to(device)
    lr_image = lr_image.unsqueeze(0)
    sr_image = gen(lr_image)
    sr_image2 = gen2(lr_image)
    sr_image = sr_image.cpu()
    sr_image2 = sr_image2.cpu()
    save_image(sr_image, "gen_img.png")
    save_image(sr_image2, "gen_img2.png")
    save_image(hr_image, "original.png")
    save_image(lr_image, "low_res.png")
