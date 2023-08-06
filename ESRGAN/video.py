from PIL import Image
from model import RRDBNet
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
import cv2


def extract_frames(gif_path, output_path):
    gif = Image.open(gif_path)
    # Check if the GIF is animated
    if gif.is_animated:
        frame_number = 0
        try:
            while True:
                gif.seek(frame_number)
                frame = gif.copy()
                frame.save(f"{output_path}/frame_{frame_number}.png", "PNG")
                frame_number += 1
        except EOFError:
            # Reached the end of the animated GIF
            pass
    else:
        # The GIF is not animated, save the single frame
        gif.save(f"{output_path}/frame_0.png", "PNG")


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


# Step 1: Read all the image files from the folder
def get_image_paths(folder_path):
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                   file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    return image_paths


def preprocess_image(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Apply any necessary transformations based on the model's input requirements
    transform4 = transforms.Compose([
        transforms.Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))
    ])
    preprocessed_image = transform4(image)
    return preprocessed_image


def load_model(model_path, device):
    gen = RRDBNet(3, 3, 64, 32, 2, 0.2).to(device)
    gen = load_weights(model_path, gen)
    gen.eval()  # Set the model to evaluation mode
    return gen


def create_gif(frames_folder, output_gif_path, duration=100):
    # frames_folder: Path to the folder containing individual frames as PNG images
    # output_gif_path: Path to save the output GIF
    # duration: The duration (in milliseconds) of each frame in the final GIF

    # Get the list of image file names in the frames folder
    image_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.png')])

    # Open all frames and store them in a list
    frames = [Image.open(os.path.join(frames_folder, file)) for file in image_files]

    # Save the frames as an animated GIF
    frames[0].save(output_gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)


if __name__ == "__main__":
    extract_frames("../SRGAN/data/HR/HR_test/giphy2.gif", "../SRGAN/data/HR/HR_test/frames/original")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = get_image_paths("../SRGAN/data/HR/HR_test/frames/original")

    gen = load_model("models/newdis/gen182.pth.tar", device)

    for image_path in image_paths:
        lr_image = preprocess_image(image_path)
        lr_image = lr_image.to(device)
        lr_image = lr_image.unsqueeze(0)
        image_name = os.path.basename(image_path)
        lr_path = os.path.join("../SRGAN/data/HR/HR_test/frames/low_res", image_name)
        save_image(lr_image, lr_path)
        with torch.no_grad():
            sr_image = gen(lr_image)
            sr_image = sr_image.cpu()
            sr_path = os.path.join("../SRGAN/data/HR/HR_test/frames/high_res", image_name)
            save_image(sr_image, sr_path)

    create_gif("../SRGAN/data/HR/HR_test/frames/high_res","../SRGAN/data/HR/HR_test/frames/high_res/a.gif",100)
    create_gif("../SRGAN/data/HR/HR_test/frames/low_res", "../SRGAN/data/HR/HR_test/frames/low_res/a.gif", 100)
    create_gif("../SRGAN/data/HR/HR_test/frames/original", "../SRGAN/data/HR/HR_test/frames/original/a.gif", 100)