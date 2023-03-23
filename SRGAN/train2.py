import torch
from tqdm import tqdm
from torch import nn
from loss import OtherLoss
from model import Generator, Discriminator
from dataloader import get_loader
import numpy as np
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def train_loop(loader, disc, gen, dis_opt, gen_opt, bce, other_loss):
    loop = tqdm(loader, leave=True)
    for idx, (hr_image, lr_image) in enumerate(loop):
        hr_image = hr_image.to(device)
        lr_image = lr_image.to(device)
        ###########################
        dis_opt.zero_grad()
        pred_real = disc(hr_image)
        ground_real = torch.ones_like(pred_real)
        real_loss = bce(pred_real, ground_real - 0.1 * torch.rand_like(pred_real))
        # real_loss.backward()
        #####################################
        fake_hr = gen(lr_image)
        pred_fake = disc(fake_hr.detach())
        ground_fake = torch.zeros_like(pred_fake)
        fake_loss = bce(pred_fake, ground_fake)
        # fake_loss.backward()
        #####################################
        dis_loss = real_loss + fake_loss
        dis_loss.backward(retain_graph=True)
        dis_opt.step()
        ####################################
        gen_opt.zero_grad()
        pred_fake = disc(fake_hr)
        adversarial_loss = 1e-3 * bce(pred_fake, torch.ones_like(pred_fake))
        gen_loss = other_loss(fake_hr, hr_image) + adversarial_loss
        gen_loss.backward()
        gen_opt.step()
        ######################################


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_loader("data/HR/DIV2K_train_HR", 16, True)
gen_model = Generator().to(device)
dis_model = Discriminator().to(device)
bce_criterion = nn.BCEWithLogitsLoss()
other_loss_crit = OtherLoss()

gen_opt = torch.optim.Adam(gen_model.parameters(), lr=1e-4, betas=(0.9, 0.999))

dis_opt = torch.optim.Adam(dis_model.parameters(), lr=1e-4, betas=(0.9, 0.999))

gen_model = gen_model.apply(weights_init)
disc_model = dis_model.apply(weights_init)

for epoch in range(20):
    train_loop(train_loader, disc_model, gen_model, dis_opt, gen_opt, bce_criterion, other_loss_crit)
    save_checkpoint(gen_model, gen_opt, filename="gen.pth.tar")
    save_checkpoint(disc_model, dis_opt, filename="dis.pth.tar")
