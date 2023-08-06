from model_or import Generator
from model import generator
import torch
from data_loader import get_loader
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import cv2
from sklearn.preprocessing import MinMaxScaler
from torchvision.transforms import InterpolationMode
from scipy.stats import norm


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#gen = Generator().to(device)
gen = generator(3, 64, 16).to(device)

gen = load_weights("gen.pth.tar", gen)
#gen2 = load_weights("gen.pth.tar", gen2)

valid_loader = get_loader("../SRGAN/data/HR/DIV2K_valid_HR", 300, 4, "Test", 1, True)
gen.eval()
#gen2.eval()

high_res, low_res = next(iter(valid_loader))
hr_image = high_res.to(device)
lr_image = low_res.to(device)
with torch.no_grad():
    u_img = gen(lr_image)
    #u_img2 = gen2(lr_image)
    u_img = u_img.cpu()
    #u_img2 = u_img2.cpu()
    hr_image = hr_image.cpu()
    lr_image = lr_image.cpu()
    save_image(u_img, "gen_img.png")
    #save_image(u_img2, "gen_img2.png")
    save_image(lr_image, "low_res.png")
    save_image(hr_image, "original.png")
