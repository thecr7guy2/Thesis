from model import RRDBNet
from uncertainity.model import DRRRDBNet
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

scaler = MinMaxScaler(feature_range=(0, 1))


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
# gen = RRDBNet2(3, 3, 64, 32, 23, 4).to(device)
# gen = RRDBNet(3, 3, 64, 32, 2, 0.2).to(device)
gen2 = DRRRDBNet(3, 3, 64, 32, 2, 0.2).to(device)
#gen2 = load_weights("bestgen.pth.tar", gen2)
gen2 = load_weights("../Infrence/models/mCD/ESRGAN/gen174.pth.tar", gen2)

# gen2.eval()
# enable_dropout(gen2)

transform4 = transforms.Compose([
    transforms.Resize((64, 64),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
])

transform5 = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
])

random_image = random.choice(os.listdir("../SRGAN/data/HR/HR_test/test/"))
image = Image.open("../SRGAN/data/HR/HR_test/test/" + random_image)
height, width = image.size
lr_image = transform4(image)
ors_image = transform5(image)

# random_image = random.choice(os.listdir("../SRGAN/data/HR/HR_test/test/"))
# image = Image.open("../SRGAN/data/HR/HR_test/test/" + random_image)
# # height, width = image.size
# # print(width, height)
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

# ghj = transform1(image)
# lr_image = transform3(ghj)
# hr_image = transform2(ghj)


lr_image = lr_image.to(device)
lr_image = lr_image.unsqueeze(0)
#print(lr_image.shape)
save_image(lr_image, "../Infrence/low_res.png")

ors_image = ors_image.unsqueeze(0)
ors_image = ors_image.to(device)
save_image(ors_image, "../Infrence/original_sized.png")

batch_list = []
for i in range(10):
    gen2.eval()
    enable_dropout(gen2)
    with torch.no_grad():
        sr_image = gen2(lr_image)
    # shape of sr_image(1,3,256,256)
    batch_list.append(sr_image)
# After 20 forward passes of the batch
stacked_batch = torch.stack(batch_list, dim=0)
# shape of the stacked batch (10,1,3,256,256)
mean_batch = torch.mean(stacked_batch, dim=0)
var_batch = torch.var(stacked_batch, dim=0)
std_batch = torch.std(stacked_batch, dim=0)

error_image = torch.abs(torch.sub(ors_image,mean_batch))

v_min, v_max = var_batch.min(), var_batch.max()
s_min, s_max = std_batch.min(), std_batch.max()
new_min, new_max = 0, 1

# print(var_batch.max())

v_p = (var_batch - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
s_p = (std_batch - s_min)/(s_max - s_min)*(new_max - new_min) + new_min
# print(torch.max(v_p))
# print(torch.max(std_batch))
# print(torch.min(std_batch))
# print(torch.max(var_batch))
# print(torch.min(var_batch))

save_image(mean_batch, "../Infrence/gen_img.png")
save_image(v_p, "../Infrence/variance_scaled.png")
save_image(std_batch, "../Infrence/std.png")
save_image(s_p, "../Infrence/std_scaled.png")


err_flat = torch.flatten(error_image)
std_flat = torch.flatten(std_batch)
ors_flat = torch.flatten(ors_image)
mean_flat = torch.flatten(mean_batch)

print(err_flat.shape)
print(std_flat.shape)

plt.scatter(std_flat.cpu().numpy(), err_flat.cpu().numpy())
plt.title("Error plot")
plt.xlabel("Standard deviation")
plt.ylabel("Error")
plt.savefig('../Infrence/error_plot.png', bbox_inches='tight')


# img = cv2.imread('../Infrence/gen_img2.png')
# img[:, :, 0] = 0
# img[:, :, 1] = 0
# cv2.imwrite("../Infrence/rchannel.png", img)
#
# img = cv2.imread('../Infrence/gen_img2.png')
# img[:, :, 1] = 0
# img[:, :, 2] = 0
# cv2.imwrite("../Infrence/bchannel.png", img)
#
# img = cv2.imread('../Infrence/gen_img2.png')
# img[:, :, 0] = 0
# img[:, :, 2] = 0
# cv2.imwrite("../Infrence/gchannel.png", img)

EPSILON = 1e-5

def confidence_interval_accuracy(y_intervals, y_true):
    interval_min, interval_max = y_intervals
    indicator = np.logical_and(y_true >= interval_min, y_true <= interval_max)

    return np.mean(indicator)

def regressor_calibration_curve(y_pred, y_true, y_std, num_points=20, distribution="gaussian"):
    """
        Computes the reliability plot for a regression prediction.
        :param y_pred: model predictions, usually the mean of the predicted distribution.
        :param y_std: model predicted confidence, usually the standard deviation of the predicted distribution.
        :param y_true: ground truth labels.
    """
    alphas = np.linspace(0.0 + EPSILON, 1.0 - EPSILON, num_points + 1)
    curve_conf = []
    curve_acc = []

    for alpha in alphas:
        alpha_intervals = norm.interval(alpha, y_pred, y_std)
        acc = confidence_interval_accuracy(alpha_intervals, y_true)

        curve_conf.append(alpha)
        curve_acc.append(acc)

    return np.array(curve_conf), np.array(curve_acc)

