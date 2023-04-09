import torch
from tqdm import tqdm
from torch import nn
from loss import OtherLoss
from model import Generator, Discriminator
from dataloader import get_loader
import numpy as np
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_loader("data/HR/DIV2K_train_HR", 16, True)
val_loader = get_loader("data/HR/DIV2K_valid_HR", 16, False)

gen_model = Generator().to(device)
dis_model = Discriminator().to(device)
bce_criterion = nn.BCEWithLogitsLoss()
psnr = PeakSignalNoiseRatio(data_range=1.0)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
other_loss_crit = OtherLoss()

gen_opt = torch.optim.Adam(gen_model.parameters(), lr=1e-4, betas=(0.9, 0.999))

dis_opt = torch.optim.Adam(dis_model.parameters(), lr=1e-4, betas=(0.9, 0.999))

gen_model = gen_model.apply(weights_init)
disc_model = dis_model.apply(weights_init)

for epoch in range(60):
    running_gen_loss = 0
    running_dis_loss = 0
    running_psnr = 0
    running_ssim = 0
    best_score = 0

    loop = tqdm(train_loader)
    gen_model.train()
    dis_model.train()
    for idx, (hr_image, lr_image) in enumerate(loop):
        hr_image = hr_image.to(device)
        lr_image = lr_image.to(device)
        ###########################
        dis_opt.zero_grad()
        pred_real = disc_model(hr_image)
        ground_real = torch.ones_like(pred_real)
        real_loss = bce_criterion(pred_real, ground_real - 0.1 * torch.rand_like(pred_real))
        # real_loss.backward()
        #####################################
        fake_hr = gen_model(lr_image)
        pred_fake = disc_model(fake_hr.detach())
        ground_fake = torch.zeros_like(pred_fake)
        fake_loss = bce_criterion(pred_fake, ground_fake)
        # fake_loss.backward()
        #####################################
        dis_loss = real_loss + fake_loss
        dis_loss.backward()
        dis_opt.step()
        ####################################
        gen_opt.zero_grad()
        pred_fake = disc_model(fake_hr)
        adversarial_loss = 0.001 * bce_criterion(pred_fake, torch.ones_like(pred_fake))
        gen_loss = other_loss_crit(fake_hr, hr_image) + adversarial_loss
        gen_loss.backward()
        gen_opt.step()
        ######################################
        running_gen_loss = running_gen_loss + gen_loss.item()
        running_dis_loss = running_dis_loss + dis_loss.item()

    epoch_gen_loss = running_gen_loss / len(train_loader)
    epoch_dis_loss = running_dis_loss / len(train_loader)
    loop.set_description(desc='[{}/{}] Loss_D: {} Loss_G: {}'.format(epoch, 60, epoch_dis_loss, epoch_gen_loss))

    gen_model.eval()

    with torch.no_grad():
        val_loop = tqdm(val_loader)
        for idx, (hr_image, lr_image) in enumerate(val_loop):
            hr_image = hr_image.to(device)
            lr_image = lr_image.to(device)
            sr_image = gen_model(lr_image)
            p_score = psnr(sr_image, hr_image)
            s_score = ssim(sr_image, hr_image)
            running_psnr = running_psnr + p_score.item()
            running_ssim = running_ssim + s_score.item()
        epoch_psnr = running_psnr / len(val_loader)
        epoch_ssim = running_ssim / len(val_loader)

        val_loop.set_description(desc='[{}/{}] SSIM: {} PSNR: {}'.format(epoch, 60, epoch_ssim, epoch_psnr))
        if epoch_psnr > best_score:
            best_score = epoch_psnr


