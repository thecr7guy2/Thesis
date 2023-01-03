import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import getdata
from Modules.generator import generator
from Modules.discriminator import discriminator
from tqdm import tqdm
from torchvision.utils import save_image

### CHANGE HERE ####
####################
learning_rate = 0.00000325
num_epochs = 8
noise_dim = 16
gen_hidden_dim = 32
dis_hidden_dim = 16
image_dim = 1
beta_1 = 0.5
beta_2 = 0.999
####################


def check_gen_images(gen, noise_dim, batch_size, device, epoch):
    gen_model.eval()
    dis_model.eval()
    with torch.no_grad():
        noise_vec = torch.randn(batch_size, noise_dim, device=device)
        t = torch.ones(batch_size) * 4
        t = t.long()
        test = one_hot(t, 10)
        test = test.to(device)
        up_noise_t = concat_vec(noise_vec, test)
        gen_image = gen(up_noise_t)
        save_image(gen_image.view(gen_image.size(0), 1, 28, 28), 'gen_images/sample_' + str(epoch) + '.png')

    gen_model.train()
    dis_model.train()


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


# creating a method to onehot encode the labels
def one_hot(labels, n_classes):
    return F.one_hot(labels, n_classes)


###################
### test method ###
# labels = torch.arange(8)  # 8 because of the batch size
# n_classes = 10
# test = one_hot(labels, n_classes)
# print(test)
####################

# Create a method to combine the noise vectors and the one hot vector
def concat_vec(vec1, vec2):
    return torch.cat((vec1.float(), vec2.float()), 1)


###################
### test method ###
# labels = torch.arange(8)  # 8 because of the batch size
# n_classes = 10
# one = one_hot(labels, n_classes)
# noise = torch.randn(8, 64)
# test = concat_vec(noise, one)
# #print(test)
# print(test[0].shape)
####################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = getdata()


# before initializing the components we have seen that we concatenate one hot encoded labels to the inputs of both the
# components. So we create a function to make changes to the input parameters of the components.

def change_input_params(noise_dim, image_dim, n_classes):
    updated_noise_dim = noise_dim + n_classes
    updated_image_dim = image_dim + n_classes
    return updated_noise_dim, updated_image_dim


up_noise_dim, up_image_dim = change_input_params(noise_dim, image_dim, 10)  # change the hardcoded classes

gen_model = generator(up_noise_dim, gen_hidden_dim, image_dim).to(device)
dis_model = discriminator(up_image_dim, dis_hidden_dim).to(device)

criterion = nn.BCEWithLogitsLoss()

gen_opt = torch.optim.Adam(gen_model.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
dis_opt = torch.optim.Adam(dis_model.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

gen_model.apply(weights_init)
dis_model.apply(weights_init)

# Initializing arrays to store the losses
gl = []
dl = []

for epoch in range(num_epochs):
    running_gen_loss = 0
    running_dis_loss = 0

    loop = tqdm(train_loader)

    gen_model.train()
    dis_model.train()

    for batch_index, (imgs, label) in enumerate(loop):
        batch_size = len(imgs)
        imgs = imgs.to(device)
        label = label.to(device)

        oh_labels = one_hot(label, 10)
        # change the hardcoded class size
        image_one_hot_labels = oh_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, 28, 28)

        # print(image_one_hot_labels.shape)
        dis_opt.zero_grad()
        noise_vec = torch.randn(batch_size, noise_dim, device=device)

        # before passing the input to the generator we concatenate both the vectors.
        up_noise_vec = concat_vec(noise_vec, oh_labels)
        fake_imgs = gen_model(up_noise_vec)

        # Now we have a set of fake images and real images but they cant still be passed as the input to the dis without
        # concatenating the one hot encoded labels

        fake_imgs_labels = concat_vec(fake_imgs.detach(), image_one_hot_labels)
        real_imgs_labels = concat_vec(imgs, image_one_hot_labels)

        fake_preds = dis_model(fake_imgs_labels)
        real_preds = dis_model(real_imgs_labels)

        ground_fakes = torch.zeros_like(fake_preds)
        ground_real = torch.ones_like(real_preds)

        fake_loss = criterion(fake_preds, ground_fakes)
        real_loss = criterion(real_preds, ground_real)

        dis_loss = (fake_loss + real_loss) / 2
        dis_loss.backward(retain_graph=True)
        dis_opt.step()

        gen_opt.zero_grad()

        fake_img_labels2 = concat_vec(fake_imgs, image_one_hot_labels)
        # Here instead of detaching we use the gradients too ?
        fake_preds2 = dis_model(fake_img_labels2)
        gen_loss = criterion(fake_preds2, torch.ones_like(fake_preds2))
        gen_loss.backward()
        gen_opt.step()

        running_gen_loss = running_gen_loss + gen_loss.item()
        running_dis_loss = running_dis_loss + dis_loss.item()

        if batch_index == 23 and epoch % 2:
            check_gen_images(gen_model, noise_dim, batch_size, device, epoch)

    epoch_gen_loss = running_gen_loss / len(train_loader)
    epoch_dis_loss = running_dis_loss / len(train_loader)

    gl.append(epoch_gen_loss)
    dl.append(epoch_dis_loss)

    print('The loss of the generator is {} and the loss of the discriminator is {} for epoch no. {}'.format(
        epoch_gen_loss, epoch_dis_loss, epoch))


