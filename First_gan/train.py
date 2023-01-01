import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import numpy as np
from dataloader import getdata
from modules.generator import generator,gen_loss
from modules.discriminator import discriminator,dis_loss
from tqdm import tqdm
from torchvision.utils import save_image


###### PARAMS ######
### CHANGE HERE ####
####################
learning_rate = 1e-4
num_epochs = 100
noise_dim = 64
hidden_dim = 128
image_dim = 784
#####################

def check_gen_images(gen,noise_dim,batch_size,device,epoch):
    gen_model.eval()
    dis_model.eval()
    with torch.no_grad():
        noise_vec = torch.normal(0,1, size=(batch_size,noise_dim))
        noise_vec = noise_vec.to(device)
        gen_image = gen(noise_vec)
        save_image(gen_image.view(gen_image.size(0), 1, 28, 28), 'gen_images/sample_' + str(epoch) + '.png')
        
    gen_model.train()
    dis_model.train()
        


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader,test_loader = getdata()


gen_model = generator(noise_dim,hidden_dim,image_dim).to(device)
dis_model = discriminator(image_dim,hidden_dim).to(device)

criterion = nn.BCEWithLogitsLoss()

gen_opt = torch.optim.Adam(gen_model.parameters(), lr=learning_rate)

dis_opt = torch.optim.Adam(dis_model.parameters(), lr=learning_rate)


gl=[]
dl=[]


for epoch in range(num_epochs):
    running_gen_loss = 0
    running_dis_loss = 0
    
    loop=tqdm(train_loader)
    
    gen_model.train()
    dis_model.train()
    
    for batch_index,(img,label) in enumerate(loop):
        batch_size = len(img)
        
        img_flatten= img.view(batch_size, -1).to(device)
        
        dis_opt.zero_grad()
        
        disc_loss = dis_loss(gen_model,dis_model,criterion,img_flatten,noise_dim,device)
        
        disc_loss.backward(retain_graph=True)
        
        dis_opt.step()
        
        
        
        gen_opt.zero_grad()
        
        gene_loss=gen_loss(gen_model,dis_model,criterion,img_flatten,noise_dim,device)
        
        gene_loss.backward(retain_graph=True)
        
        gen_opt.step()
        
        
        
        running_gen_loss = running_gen_loss + gene_loss.item()
        running_dis_loss = running_dis_loss + disc_loss.item()
        
        if batch_index == 23 and epoch % 2:
            check_gen_images(gen_model,noise_dim,batch_size,device,epoch)
        
    epoch_gen_loss=running_gen_loss/len(train_loader)
    epoch_dis_loss=running_dis_loss/len(train_loader)
    
    gl.append(epoch_gen_loss)
    dl.append(epoch_dis_loss)
    
    print('The loss of the generator is {} and the loss of the discriminator is {} for epoch no. {}'.format(epoch_gen_loss, epoch_dis_loss , epoch))


with torch.no_grad():
        noise_vec = torch.normal(0,1, size=(batch_size*2,noise_dim))
        noise_vec = noise_vec.to(device)
        gen_image = gen_model(noise_vec)
        save_image(gen_image.view(batch_size*2, 1, 28, 28), 'gen_images/final_image.png')        
        

        
        
        


