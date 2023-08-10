from PIL import Image, ImageDraw
import random


def canh(path1, path2, cropsize):
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    seed_x = 32
    seed_y = 77
    # Take first random crop
    random.seed(seed_x)
    x1 = random.randint(0, img1.size[0] - cropsize)
    y1 = random.randint(0, img1.size[1] - cropsize)
    crop1_x = img1.crop((x1, y1, x1 + cropsize, y1 + cropsize))
    crop2_x = img2.crop((x1, y1, x1 + cropsize, y1 + cropsize))

    # Take second random crop
    random.seed(seed_y)
    x2 = random.randint(0, img1.size[0] - cropsize)
    y2 = random.randint(0, img1.size[1] - cropsize)
    crop1_y = img1.crop((x2, y2, x2 + cropsize, y2 + cropsize))
    crop2_y = img2.crop((x2, y2, x2 + cropsize, y2 + cropsize))

    # Highlight crops on original image
    draw = ImageDraw.Draw(img1)
    draw.rectangle([(x1, y1), (x1 + cropsize, y1 + cropsize)], outline='green', width=4)
    # draw.rectangle([(x2, y2), (x2 + cropsize, y2 + cropsize)], outline='red', width=8)

    # Save output images
    crop1_x.save('../Images/Results/or_crop1.png')
    crop2_x.save('../Images/Results/sr_crop1.png')
    crop1_y.save('../Images/Results/or_crop2.png')
    crop2_y.save('../Images/Results/sr_crop2.png')
    img1.save('../Images/Results/or_high.jpg')
    
    
size = 64
canh("../Images/Datasets/Set14/GTmod12/ppt3.png", "../Images/Results/gen_img.png", size)
