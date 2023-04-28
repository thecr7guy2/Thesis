from PIL import Image
import os

directory = "DIV2K_train_HR"

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    im = Image.open(f)
    w, h = im.size
    if w*h < 65536:
        os.remove(f)
