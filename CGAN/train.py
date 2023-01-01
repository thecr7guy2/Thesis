import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import getdata
from Modules.generator import generator
from Modules.discriminator import discriminator

### CHANGE HERE ####
####################
learning_rate = 0.00005
num_epochs = 8
noise_dim = 16
gen_hidden_dim = 64
dis_hidden_dim = 16
image_dim = 1
beta_1 = 0.5
beta_2 = 0.999


#####################


def check_gen_images(gen, noise_dim, batch_size, device, epoch):
    gen_model.eval()
    dis_model.eval()
    with torch.no_grad():
        noise_vec = torch.randn(batch_size, noise_dim, device=device)
        gen_image = gen(noise_vec)
        save_image(gen_image.view(gen_image.size(0), 1, 28, 28), 'gen_images/sample_' + str(epoch) + '.png')

    gen_model.train()
    dis_model.train()


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
# inputs of the components. So we create a function to make changes to the input parameters of the components.

def change_input_params(noise_dim, image_dim, n_classes):
    updated_noise_dim = noise_dim + n_classes
    updated_image_dim = image_dim + n_classes
    return updated_noise_dim, updated_image_dim


up_noise_dim, up_image_dim = change_input_params(noise_dim, image_dim, 10)  # change the hardcoded classes

gen_model = generator(up_noise_dim, gen_hidden_dim, image_dim).to(device)
dis_model = discriminator(up_image_dim, dis_hidden_dim).to(device)
