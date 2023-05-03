import torch
from torch import nn
from model import RRDBNet, Discriminator
from tqdm import tqdm
from data_loader import get_loader
from loss import PerceptualLoss
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
import os

experiment = "ESRGAN_dis1"
exp_path = os.path.join("models", experiment)
os.mkdir(exp_path)


def save_checkpoint(model, optimizer, scheduler, epoch, psnr, ssim, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        'epoch': epoch,
        "psnr": psnr,
        "ssim": ssim,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, scheduler):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    curr_epoch = checkpoint["epoch"]
    ssim = checkpoint["ssim"]
    psnr = checkpoint["psnr"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    lr_epoch = scheduler.get_last_lr()
    lr_epoch = lr_epoch[0]
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_epoch
    return model, optimizer, scheduler, curr_epoch, psnr, ssim


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


def gen_weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        m.weight.data *= 0.1
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def dis_weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)


def train_one_epoch(gen_model, dis_model, loader, pcriterion, acriterion, l1criterion, gen_op, dis_op):
    running_gen_loss = 0
    running_dis_loss = 0
    running_true_prob = 0
    running_fake_prob = 0
    loop = tqdm(loader)
    gen_model.train()
    dis_model.train()
    for idx, (hr_image, lr_image) in enumerate(loop):
        hr_image = hr_image.to(device)
        lr_image = lr_image.to(device)
        batch_size, _, _, _ = hr_image.shape
        real_labels = torch.full([batch_size, 1], 1.0, dtype=hr_image.dtype, device=device)
        fake_labels = torch.full([batch_size, 1], 0.0, dtype=hr_image.dtype, device=device)
        ##########################################
        # Starting the gen training so turn off dis backprop
        for d_parameters in dis_model.parameters():
            d_parameters.requires_grad = False
        gen_model.zero_grad(set_to_none=True)
        fake_hr = gen_model(lr_image)
        pred_real = dis_model(hr_image.detach().clone())
        pred_fake = dis_model(fake_hr)
        pixel_loss = 0.01 * l1criterion(fake_hr, hr_image)
        perceptual_loss = pcriterion(fake_hr, hr_image)
        loss1 = acriterion(pred_real - torch.mean(pred_fake), fake_labels) * 0.5
        loss2 = acriterion(pred_fake - torch.mean(pred_real), real_labels) * 0.5
        adversarial_loss = 0.005 * (loss1 + loss2)
        gen_loss = pixel_loss + perceptual_loss + adversarial_loss
        gen_loss.backward()
        gen_op.step()
        ###########################################################
        # Generator training done. We can now unfreeze the Dis
        for d_parameters in dis_model.parameters():
            d_parameters.requires_grad = True
        dis_model.zero_grad(set_to_none=True)
        pred_real = dis_model(hr_image)
        pred_fake = dis_model(fake_hr.detach().clone())
        real_loss = acriterion(pred_real - torch.mean(pred_fake), real_labels) * 0.5
        real_loss.backward(retain_graph=True)
        ###########################################################
        pred_fake = dis_model(fake_hr.detach().clone())
        fake_loss = acriterion(pred_fake - torch.mean(pred_real), fake_labels) * 0.5
        fake_loss.backward()
        dis_loss = real_loss + fake_loss
        dis_op.step()
        d_gt_probability = torch.sigmoid_(torch.mean(pred_real.detach()))
        d_sr_probability = torch.sigmoid_(torch.mean(pred_fake.detach()))
        ##############################################################
        running_gen_loss = running_gen_loss + gen_loss.item()
        running_dis_loss = running_dis_loss + dis_loss.item()
        running_true_prob = running_true_prob + d_gt_probability.item()
        running_fake_prob = running_fake_prob + d_sr_probability.item()

    return running_gen_loss, running_dis_loss, running_true_prob, running_fake_prob


def eval_one_epoch(gen_model, loader, psnr_criterion, ssim_criterion):
    running_psnr = 0
    running_ssim = 0
    loop = tqdm(loader)
    gen_model.eval()
    with torch.no_grad():
        for idx, (hr_image, lr_image) in enumerate(loop):
            hr_image = hr_image.to(device)
            lr_image = lr_image.to(device)
            sr_image = gen_model(lr_image)
            psnr = psnr_criterion(sr_image, hr_image)
            ssim = ssim_criterion(sr_image, hr_image)

            running_psnr = running_psnr + psnr.item()
            running_ssim = running_ssim + ssim.item()

        return running_psnr, running_ssim


epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_loader("../SRGAN/data/HR/DIV2K_train_HR", 16, True)
val_loader = get_loader("../SRGAN/data/HR/DIV2K_valid_HR", 16, False)
print("Load all datasets successfully.\n")

gen = RRDBNet(3, 3, 64, 32, 2, 0.2).to(device)
print("Loaded the Generator successfully.\n")

dis = Discriminator().to(device)
print("Loaded the Discriminator successfully.\n")

perceptual_crit = PerceptualLoss()
perceptual_crit = perceptual_crit.to(device)

l1_crit = nn.L1Loss()
l1_crit = l1_crit.to(device)

adversarial_crit = nn.BCEWithLogitsLoss()
adversarial_crit = adversarial_crit.to(device)
print("Defined loss functions successfully.\n")

gen_opt = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
dis_opt = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
print("Defined all optimizer functions successfully.\n")

model_mile = [int(epochs * 0.125), int(epochs * 0.250), int(epochs * 0.500), int(epochs * 0.750)]

gen_scheduler = torch.optim.lr_scheduler.MultiStepLR(gen_opt, model_mile, 0.5)
dis_scheduler = torch.optim.lr_scheduler.MultiStepLR(dis_opt, model_mile, 0.5)
print("Defined all optimizer scheduler functions successfully.\n")

psnr_crit = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_crit = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

print("Checking if the user wants to continue training\n")
ui = int(input('''Press 1 to load pre-trained weights.
                Press 2 to start training the models from scratch\n'''))
if ui == 1:
    gen, gen_opt, gen_scheduler, curr_epoch, psnr, ssim = load_checkpoint("gen.pth.tar", gen, gen_opt, gen_scheduler)
    dis, dis_opt, dis_scheduler, curr_epoch2, psnr2, ssim2 = load_checkpoint("dis.pth.tar", gen, gen_opt, gen_scheduler)
    print("Checkpoint Loaded Successfully. Training will now Resume\n")
    print("The last best psnr recorded was" + str(psnr))
    start = curr_epoch + 1
    best_psnr = psnr
    best_ssim = ssim
else:
    gen = load_weights("bestgen.pth.tar", gen)
    #gen = gen.apply(gen_weights_init)
    #dis = dis.apply(dis_weights_init)
    print("The model will now be trained from scratch\n")
    start = 0
    best_psnr = 0
    best_ssim = 0

for epoch in range(start, epochs):
    a, b, e, f = train_one_epoch(gen, dis, train_loader, perceptual_crit, adversarial_crit, l1_crit, gen_opt, dis_opt)
    epoch_gen_loss = a / len(train_loader)
    epoch_dis_loss = b / len(train_loader)
    epoch_true_prob = e / len(train_loader)
    epoch_fake_prob = f / len(train_loader)
    c, d = eval_one_epoch(gen, val_loader, psnr_crit, ssim_crit)
    epoch_psnr = c / len(val_loader)
    epoch_ssim = d / len(val_loader)
    dis_scheduler.step()
    gen_scheduler.step()
    with open("train_log_" + experiment + ".txt", "a") as f:
        f.write(
            '[{}/{}] GEN_LOSS:{}, DIS_LOSS: {} True_prob: {} Fake_prob: {} PSNR: {} SSIM: {} \n'.format(epoch, epochs,
                                                                                                        epoch_gen_loss,
                                                                                                        epoch_dis_loss,
                                                                                                        epoch_true_prob,
                                                                                                        epoch_fake_prob,
                                                                                                        epoch_psnr,
                                                                                                        epoch_ssim))
    f.close()
    if epoch_psnr > best_psnr and epoch_ssim > best_ssim:
        best_psnr = epoch_psnr
        best_ssim = epoch_ssim
        save_checkpoint(gen, gen_opt, gen_scheduler, epoch, best_psnr, best_ssim,
                        filename="models/" + experiment + "/gen" + str(epoch) + ".pth.tar")
        save_checkpoint(dis, dis_opt, dis_scheduler, epoch, best_psnr, best_ssim,
                        filename="models/" + experiment + "/dis" + str(epoch) + ".pth.tar")

    if (epoch == epochs - 1) or (epoch % 5 == 0):
        save_checkpoint(gen, gen_opt, gen_scheduler, epoch, best_psnr, best_ssim,
                        filename="models/" + experiment + "/gen" + str(epoch) + ".pth.tar")
        save_checkpoint(dis, dis_opt, dis_scheduler, epoch, best_psnr, best_ssim,
                        filename="models/" + experiment + "/dis" + str(epoch) + ".pth.tar")
