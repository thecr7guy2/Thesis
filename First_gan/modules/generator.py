import torch
import torch.nn as nn 
from torchvision.utils import save_image



class generator(nn.Module):
    """
    The following module generates an image when given a random noise as an input
    Parameters:
        input_dim:
        hidden_dim:
        im_dim:
        noise:
    """
    
    
    
    def __init__(self,input_dim,hidden_dim,im_dim):
        super(generator, self).__init__()
        self.generator = nn.Sequential(
            self.generator_block(input_dim,hidden_dim),
            self.generator_block(hidden_dim,hidden_dim*2),
            self.generator_block(hidden_dim*2,hidden_dim*4),
            self.generator_block(hidden_dim*4,hidden_dim*8),
            nn.Linear(hidden_dim*8,im_dim),
            nn.Sigmoid() 
        )
        
        
    def generator_block(self,input,output):
        block= nn.Sequential(
            nn.Linear(input,output),
            nn.BatchNorm1d(output),
            nn.ReLU(inplace=True))
        return block 
   

    def forward(self, noise):
        
        xd= self.generator(noise)

        return xd
    
    
    
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# gen = generator(64,128,784).to(device)
# noise= torch.normal(0,1, size=(2, 64)).to(device)

# gen_image=gen(noise)
# epoch=10

# save_image(gen_image.view(gen_image.size(0), 1, 28, 28), '../gen_images/sample_' + str(epoch) + '.png')

# print(gen_image.shape)

def gen_loss(gen,disc,criterion,real_im,noise_dim,device):
    
    noise_vec = torch.normal(0,1, size=(len(real_im),noise_dim))
    noise_vec = noise_vec.to(device)
    
    fake_images= gen(noise_vec)
    
    pred_fakes = disc(fake_images)
    
    ground_real = torch.ones_like(pred_fakes)
    
    gen_loss=criterion(pred_fakes,ground_real)
    
    return gen_loss
    
    
    