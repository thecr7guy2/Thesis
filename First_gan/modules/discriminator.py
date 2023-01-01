import torch
import torch.nn as nn 

class discriminator(nn.Module):
    """Some Information about discriminator"""
    def __init__(self,im_dim,hidden_dim):
        super(discriminator, self).__init__()
        self.dis= nn.Sequential(
            self.dis_block(im_dim,hidden_dim*4),
            self.dis_block(hidden_dim*4,hidden_dim*2),
            self.dis_block(hidden_dim*2,hidden_dim),
            self.dis_block(hidden_dim,32),
            nn.Linear(32,1)
        )
        
        
    def dis_block(self,in_dim,out_dim):
        block= nn.Sequential(
            nn.Linear(in_dim,out_dim),
            nn.LeakyReLU(0.2,inplace=True)
        )
        return block

    def forward(self, image):
        
        x = self.dis(image)

        return x
    
    
def dis_loss(gen,disc,criterion,real_im,noise_dim,device):
    
    noise_vec = torch.normal(0,1, size=(len(real_im),noise_dim))
    noise_vec = noise_vec.to(device)
    
    #pass the noise as an input to the generator to generate fake images
    
    fake_images= gen(noise_vec)
    
    #Now use the discriminator and get the predcitions of the generated fake images
    
    pred_fakes = disc(fake_images.detach())
    
    #Create the ground truth vector
    
    ground_fakes = torch.zeros_like(pred_fakes)
    
    #Ue the costfunction to compute the loss caused by fake images
    
    fake_loss=criterion(pred_fakes,ground_fakes)
    
    #repeat similar process for the real images
    
    pred_real =disc(real_im)
    
    #Create the ground truth vector
    
    ground_real = torch.ones_like(pred_real)
    
    real_loss = criterion(pred_real,ground_real)
    
    disc_loss = (real_loss + fake_loss) / 2
    
    return disc_loss
    
    
    
    
    
    
# batch_size=2
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# dis = discriminator(784,128).to(device)
# images= torch.normal(0,1, size=(batch_size,1,28,28))
# print(images.shape)
# images = images.view(batch_size, -1).to(device)
# print(images.shape)
# print("########################")
# pred= dis(images)
# print(pred.shape)
# print(pred)


