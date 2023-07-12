from model import DRRRDBNet
import torch
from data_loader import get_loader
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure


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


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen_model = DRRRDBNet(3, 3, 64, 32, 2, 0.2).to(device)
psnr_crit = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_crit = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
valid_loader = get_loader("../../SRGAN/data/HR/DIV2K_valid_HR", 256, 4, "Valid", 1, True)
gen_model = load_weights("models/gen316.pth.tar", gen_model)
gen_model.eval()
enable_dropout(gen_model)

high_res, low_res = next(iter(valid_loader))
hr_image = high_res.to(device)
lr_image = low_res.to(device)
sr_list = []
with torch.no_grad():
    for i in range(20):
        sr_image = gen_model(lr_image)
        sr_list.append(sr_image)
        sr_image = sr_image.cpu()
        save_image(sr_image, "images/gen_img" + str(i) + ".png")

psnr = psnr_crit(sr_list[4], sr_list[16])
ssim = ssim_crit(sr_list[4], sr_list[16])

print(psnr)
print(ssim)