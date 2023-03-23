import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from dataloader import get_loader
from modules.generator import generator
from modules.discriminator import discriminator
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms
import wandb

wandb.init(project="DOGGan", entity="thecr7guy")

# ###################
# ### Parameters ####
# dis_learning_rate = 1e-4
# num_epochs = 25
# noise_dim = 100
# gen_hidden_dim = 64
# dis_hidden_dim = 64
# image_dim = 3
# beta_1 = 0.5
# beta_2 = 0.999
# batch_size = 64
# ####################

wandb.config = {
    "dis_lr": 0.0005,
    "gen_lr": 0.001,
    "num_epochs": 5,
    "noise_dim": 128,
    "gen_hidden_dim": 64,
    "dis_hidden_dim": 32,
    "image_dim": 3,
    "beta_1": 0.5,
    "beta_2": 0.999,
    "batch_size": 32,
}


def check_gen_images(gen, noise_dim, batch_size, device, epoch):
    gen_model.eval()
    dis_model.eval()
    with torch.no_grad():
        noise_vec = torch.randn(batch_size, noise_dim, device=device)
        gen_image = gen(noise_vec)
        save_image(gen_image.view(gen_image.size(0), 3, 64, 64), 'gen_images/sample_' + str(epoch) + '.png')
        wandb.log({"images_ep": wandb.Image('gen_images/sample_' + str(epoch) + '.png')})

    gen_model.train()
    dis_model.train()


random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=20)]
dog_transform = transforms.Compose([transforms.Resize(64),
                                    transforms.CenterCrop(64),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomApply(random_transforms, p=0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader = get_loader("data/images", dog_transform, wandb.config["batch_size"], shuffle=True)

gen_model = generator(wandb.config["noise_dim"], wandb.config["gen_hidden_dim"], wandb.config["image_dim"]).to(device)
dis_model = discriminator(wandb.config["image_dim"], wandb.config["dis_hidden_dim"]).to(device)

criterion = nn.BCEWithLogitsLoss()

gen_opt = torch.optim.Adam(gen_model.parameters(), lr=wandb.config["gen_lr"],
                           betas=(wandb.config["beta_1"], wandb.config["beta_2"]))

dis_opt = torch.optim.Adam(dis_model.parameters(), lr=wandb.config["dis_lr"],
                           betas=(wandb.config["beta_1"], wandb.config["beta_2"]))

gen_model = gen_model.apply(weights_init)
disc_model = dis_model.apply(weights_init)

for epoch in range(wandb.config["num_epochs"]):
    running_gen_loss = 0
    running_dis_loss = 0

    loop = tqdm(train_loader)

    gen_model.train()
    dis_model.train()

    for batch_index, (img) in enumerate(loop):
        batch_size = len(img)
        # print(img.shape)
        # img_flatten = img.view(batch_size, -1).to(device)
        img = img.to(device)
        ############
        dis_opt.zero_grad()
        pred_real = dis_model(img)
        ground_real = torch.ones_like(pred_real)
        real_loss = criterion(pred_real, ground_real)
        real_loss.backward()
        ##############
        noise_vec = torch.randn(len(img), wandb.config["noise_dim"], device=device)
        fake_images = gen_model(noise_vec)
        pred_fakes = dis_model(fake_images.detach())
        ground_fakes = torch.zeros_like(pred_fakes)
        fake_loss = criterion(pred_fakes, ground_fakes)
        fake_loss.backward()
        ##############
        dis_loss = (real_loss + fake_loss) / 2
        dis_opt.step()
        ###########################
        noise_vec_2 = torch.randn(len(img), wandb.config["noise_dim"], device=device)
        fake_images_2 = gen_model(noise_vec_2)
        pred_fakes_2 = dis_model(fake_images_2)
        gen_loss = criterion(pred_fakes_2, ground_real)
        gen_loss.backward()
        gen_opt.step()
        #####################################
        running_gen_loss = running_gen_loss + gen_loss.item()
        running_dis_loss = running_dis_loss + dis_loss.item()

        if batch_index == 23:
            check_gen_images(gen_model, wandb.config["noise_dim"], wandb.config["batch_size"], device, epoch)

    epoch_gen_loss = running_gen_loss / len(train_loader)
    epoch_dis_loss = running_dis_loss / len(train_loader)
    print('The loss of the generator is {} and the loss of the discriminator is {} for epoch no. {}'.format(
        epoch_gen_loss, epoch_dis_loss, epoch))
    wandb.log({"gen_loss": epoch_gen_loss})
    wandb.log({"dis_loss": epoch_dis_loss})

with torch.no_grad():
    noise_vec = torch.randn(wandb.config["batch_size"] * 2, wandb.config["noise_dim"], device=device)
    gen_image = gen_model(noise_vec)
    save_image(gen_image.view(wandb.config["batch_size"] * 2, 3, 64, 64), 'gen_images/final_image.png')
