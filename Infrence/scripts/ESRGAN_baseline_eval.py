import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch.ESRGAN_arch import RRDBNet
import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import random
from torchvision.transforms import InterpolationMode


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
gen2 = RRDBNet(3, 3, 64, 32, 2, 0.2).to(device)
gen2 = load_weights("../models/ESRGAN/Ensemble/model2/gen200.pth.tar", gen2)
gen2.eval()

random_image = random.choice(os.listdir("../Images/Datasets/test/"))
image = Image.open("../Images/Datasets/test/" + random_image)
height, width = image.size
print(random_image)

transform4 = transforms.Compose([
    # transforms.RandomCrop((480, 480)),
    #transforms.Resize((width//4, height//4), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
])

lr_image = transform4(image)

with torch.no_grad():
    lr_image = lr_image.to(device)
    lr_image = lr_image.unsqueeze(0)
    # sr_image = gen(lr_image)
    sr_image2 = gen2(lr_image)
    # sr_image = sr_image.cpu()
    sr_image2 = sr_image2.cpu()
    # print(sr_image2.squeeze(0).shape)
    # save_image(sr_image, "gen_img.png")
    save_image(sr_image2, "../Images/Results/gen_img.png")
