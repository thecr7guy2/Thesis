from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

low_res = Image.open("original.png")
sup_res = Image.open("gen_img.png")


low_res = np.array(low_res)
low_res = low_res.astype(np.float32)
sup_res = np.array(sup_res).astype(np.float32)

ssim_score = ssim(low_res, sup_res, data_range=sup_res.max() - sup_res.min(),channel_axis=2)
print(ssim_score)