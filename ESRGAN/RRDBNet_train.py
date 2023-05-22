import torch
from torch import nn
from torch import optim
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
from data_loader import get_loader
from model import RRDBNet
from tqdm import tqdm
import os

experiment = "RRDB_train400"
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
    return curr_epoch, psnr, ssim


def gen_weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        m.weight.data *= 0.1
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def train_one_epoch(gen_model, loader, criterion, optimizer):
    running_loss = 0
    loop = tqdm(loader)
    gen_model.train()
    for idx, (hr_image, lr_image) in enumerate(loop):
        hr_image = hr_image.to(device)
        lr_image = lr_image.to(device)
        gen_model.zero_grad(set_to_none=True)
        sr_image = gen_model(lr_image)
        loss = torch.mul(1.0, criterion(sr_image, hr_image))
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()

    return running_loss


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


epochs = 350
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = get_loader("../SRGAN/data/HR/DIV2K_train_HR", 256, 4, "Train", 16, True)
val_loader = get_loader("../SRGAN/data/HR/DIV2K_valid_HR", 256, 4, "Valid", 16, False)
print("Load all datasets successfully.\n")

l1criterion = nn.L1Loss()
l1criterion = l1criterion.to(device)
print("Defined loss functions successfully.\n")

gen = RRDBNet(3, 3, 64, 32, 2, 0.2).to(device)
print("Loaded the RRDBNet model successfully.\n")

optimizer = optim.Adam(gen.parameters(), 2e-4, (0.9, 0.99), 1e-8, 0.0)
print("Defined all optimizer functions successfully.\n")

scheduler = optim.lr_scheduler.StepLR(optimizer, epochs // 5, 0.5)
print("Defined all optimizer scheduler functions successfully.\n")

psnr_crit = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_crit = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

print("Checking if the user wants continue training and load pretrained weights\n")
ui = int(input('''Press 1 to load pre-trained weights.
                Press 2 to start training the models from scratch\n'''))
if ui == 1:
    curr_epoch, psnr, ssim = load_checkpoint("gen.pth.tar", gen, optimizer, scheduler)
    print("Checkpoint Loaded Successfully. Training will now Resume\n")
    print("The last best psnr recorded was" + str(psnr))
    start = curr_epoch + 1
    best_psnr = psnr
    best_ssim = ssim
else:
    gen = gen.apply(gen_weights_init)
    print("The model will now be trained from scratch\n")
    start = 0
    best_psnr = 0
    best_ssim = 0

for epoch in range(start, epochs):
    a = train_one_epoch(gen, train_loader, l1criterion, optimizer)
    epoch_loss = a / len(train_loader)
    b, c = eval_one_epoch(gen, val_loader, psnr_crit, ssim_crit)
    epoch_psnr = b / len(val_loader)
    epoch_ssim = c / len(val_loader)
    scheduler.step()
    with open("train_log_" + experiment + ".txt", "a") as f:
        f.write('[{}/{}] The loss:{} SSIM: {} PSNR: {}\n'.format(epoch, 300, epoch_loss, epoch_ssim, epoch_psnr))
    f.close()
    if epoch_psnr > best_psnr and epoch_ssim > best_ssim:
        best_psnr = epoch_psnr
        best_ssim = epoch_ssim
        save_checkpoint(gen, optimizer, scheduler, epoch, best_psnr, best_ssim,
                        filename="models/" + experiment + "/gen" + str(epoch) + ".pth.tar")

    if (epoch == epochs - 1) or (epoch % 2 == 0):
        save_checkpoint(gen, optimizer, scheduler, epoch, best_psnr, best_ssim,
                        filename="models/" + experiment + "/gen" + str(epoch) + ".pth.tar")
