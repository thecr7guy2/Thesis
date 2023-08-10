from model import RRDBNet
from model3 import RRDBNet2
from uncertainity.model import DRRRDBNet
import torch
from data_loader import get_loader
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import os
import random


def load_weights(checkpoint_file, model):
    print("=> Loading weights")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                  k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
    model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict)
    print("Successfully loaded the pretrained model weights")
    return model

def load_weights2(checkpoint_file, model):
    print("=> Loading weights")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model_state_dict = model.state_dict()
    state_dict = {k: v for k, v in checkpoint.items() if
                  k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
    model_state_dict.update(state_dict)
    model.load_state_dict(model_state_dict)
    print("Successfully loaded the pretrained model weights")
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#gen = RRDBNet2(3, 3, 64, 32, 23, 4).to(device)
# gen = RRDBNet(3, 3, 64, 32, 2, 0.2).to(device)
gen2 = RRDBNet(3, 3, 64, 32, 2, 0.2).to(device)

#gen = load_weights("ESRGAN_x4-DFO2K-25393df7.pth.tar", gen)
gen2 = load_weights("models/ensemble3/gen200.pth.tar", gen2)
#valid_loader = get_loader("../SRGAN/data/HR/DIV2K_valid_HR", 256, 4, "Valid", 1, True)
#gen.eval()
gen2.eval()

# high_res, low_res = next(iter(valid_loader))
# hr_image = high_res.to(device)
# lr_image = low_res.to(device)
# with torch.no_grad():
#     u_img = gen(lr_image)
#     u_img2 = gen2(lr_image)
#     u_img = u_img.cpu()
#     u_img2 = u_img2.cpu()
#     hr_image = hr_image.cpu()
#     lr_image = lr_image.cpu()
#     save_image(u_img, "gen_img.png")
#     save_image(u_img2, "gen_img2.png")
#     save_image(lr_image, "low_res.png")
#     save_image(hr_image, "original.png")

# random_image = random.choice(os.listdir("../SRGAN/data/HR/HR_test/"))
# image = Image.open("../SRGAN/data/HR/HR_test/"+random_image)
# height, width = image.size
# print(width, height)
# transform1 = transforms.Compose([
#     transforms.RandomCrop((256, 256)),
# ])
# transform2 = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
# ])
#
# transform3 = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
# ])
#
# ghj = transform1(image)
# lr_image = transform3(ghj)
# hr_image = transform2(ghj)

# ################################################################
transform4 = transforms.Compose([
    #transforms.RandomCrop((480,480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
])

random_image = random.choice(os.listdir("../SRGAN/data/HR/HR_test/test/"))
image = Image.open("../SRGAN/data/HR/HR_test/test/"+random_image)
height, width = image.size
lr_image = transform4(image)
# # #######################################################################

with torch.no_grad():
    lr_image = lr_image.to(device)
    lr_image = lr_image.unsqueeze(0)
    #sr_image = gen(lr_image)
    sr_image2 = gen2(lr_image)
    #sr_image = sr_image.cpu()
    sr_image2 = sr_image2.cpu()
    print(sr_image2.squeeze(0).shape)
    #save_image(sr_image, "gen_img.png")
    save_image(sr_image2, "gen_img2.png")
    #save_image(hr_image, "original.png")
    save_image(lr_image, "low_res.png")

# check = torch.load("interp_08.pth.tar")
# print(type(check))