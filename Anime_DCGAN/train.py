import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from dataloader import get_loader
from modules.generator import generator, gen_loss
from modules.discriminator import discriminator, dis_loss
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms

###### PARAMS ######
### CHANGE HERE ####
####################
learning_rate = 1e-4
num_epochs = 25
noise_dim = 100
gen_hidden_dim = 64
dis_hidden_dim = 64
image_dim = 3
beta_1 = 0.01
beta_2 = 0.999
batch_size = 64
#####################

def check_gen_images(gen, noise_dim, batch_size, device, epoch):
    gen_model.eval()
    dis_model.eval()
    with torch.no_grad():
        noise_vec = torch.randn(batch_size, noise_dim, device=device)
        gen_image = gen(noise_vec)
        save_image(gen_image.view(gen_image.size(0), 3, 64,64), 'gen_images/sample_' + str(epoch) + '.png')

    gen_model.train()
    dis_model.train()


anime_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader = get_loader("data/images", anime_transform, batch_size, shuffle=True)

gen_model = generator(noise_dim, gen_hidden_dim, image_dim).to(device)
dis_model = discriminator(image_dim, dis_hidden_dim).to(device)

criterion = nn.BCEWithLogitsLoss()

gen_opt = torch.optim.Adam(gen_model.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

dis_opt = torch.optim.Adam(dis_model.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

gen_model = gen_model.apply(weights_init)
disc_model = dis_model.apply(weights_init)

gl = []
dl = []

for epoch in range(num_epochs):
    running_gen_loss = 0
    running_dis_loss = 0

    loop = tqdm(train_loader)

    gen_model.train()
    dis_model.train()

    for batch_index, (img) in enumerate(loop):
        batch_size = len(img)
        #print(img.shape)
        # img_flatten = img.view(batch_size, -1).to(device)
        img = img.to(device)

        dis_opt.zero_grad()

        disc_loss = dis_loss(gen_model, dis_model, criterion, img, noise_dim, device)

        disc_loss.backward(retain_graph=True)

        dis_opt.step()

        gen_opt.zero_grad()

        gene_loss = gen_loss(gen_model, dis_model, criterion, img, noise_dim, device)

        gene_loss.backward()

        gen_opt.step()

        running_gen_loss = running_gen_loss + gene_loss.item()
        running_dis_loss = running_dis_loss + disc_loss.item()

        if batch_index == 23 and epoch % 2:
            check_gen_images(gen_model, noise_dim, batch_size, device, epoch)

    epoch_gen_loss = running_gen_loss / len(train_loader)
    epoch_dis_loss = running_dis_loss / len(train_loader)

    gl.append(epoch_gen_loss)
    dl.append(epoch_dis_loss)

    print('The loss of the generator is {} and the loss of the discriminator is {} for epoch no. {}'.format(
        epoch_gen_loss, epoch_dis_loss, epoch))

with torch.no_grad():
    noise_vec = torch.randn(batch_size * 2, noise_dim, device=device)
    gen_image = gen_model(noise_vec)
    save_image(gen_image.view(batch_size * 2, 3,64,64), 'gen_images/final_image.png')
