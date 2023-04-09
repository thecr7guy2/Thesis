import torch.nn as nn
from torchvision.models import vgg19
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OtherLoss(nn.Module):
    def __init__(self):
        super(OtherLoss, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:31].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.mse_loss = nn.MSELoss()

    def forward(self, input_im, target_im):
        image_loss = self.mse_loss(input_im, target_im)
        perception_loss = self.mse_loss(self.vgg(input_im), self.vgg(target_im))
        return image_loss + 0.006*perception_loss
