import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch.SRGAN_arch import generator
import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = generator(3, 64, 16).to(device)
gen = load_weights("../models/SRGAN/normal/gen270.pth.tar", gen)
gen.eval()

############
random_image = random.choice(os.listdir("../Images/Datasets/test/"))
image = Image.open("../Images/Datasets/test/" + random_image)
height, width = image.size
print(random_image)
############

transform4 = transforms.Compose([
    # transforms.RandomCrop((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
])

lr_image = transform4(image)
############

with torch.no_grad():
    lr_image = lr_image.to(device)
    lr_image = lr_image.unsqueeze(0)
    # sr_image = gen(lr_image)
    sr_image2 = gen(lr_image)
    # sr_image = sr_image.cpu()
    sr_image2 = sr_image2.cpu()
    # print(sr_image2.squeeze(0).shape)
    # save_image(sr_image, "gen_img.png")
    save_image(sr_image2, "../Images/Results/gen_img.png")
    # save_image(hr_image, "original.png")
    # save_image(lr_image, "low_res.png")
