import cv2
from skimage.metrics import structural_similarity

def metrics(x1, x2):
    img1 = cv2.imread(x1)
    img2 = cv2.imread(x2)

    img1_ycbcr = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
    img2_ycbcr = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)

    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1_y = img1_ycbcr[..., 0]
    img2_y = img2_ycbcr[..., 0]

    psnr = cv2.PSNR(img1_y, img2_y)
    #ssim2 = ssim(img1_y, img2_y)
    ssim = structural_similarity(img1_y, img2_y, data_range=img1_y.max() - img1_y.min())
    # ssim2 = structural_similarity(gray1,gray2)

    return psnr,ssim


a,b = metrics("../Images/Datasets/Set14/GTmod12/ppt3.png", "../Images/Results/gen_img.png")

print(a,b)

