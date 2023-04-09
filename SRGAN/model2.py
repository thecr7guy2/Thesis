import torch
from torch import nn
from torchinfo import summary


class Encoder_block(nn.Module):
    def __init__(self, channels):
        super(Encoder_block, self).__init__()
        self.primary = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(num_parameters=64)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.PReLU(num_parameters=64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            # nn.PReLU(num_parameters=32),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(16*24*24, 4096),
            nn.Unflatten(1, torch.Size([16, 16, 16])),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=3),
        )

    def forward(self, inp):
        c = self.primary(inp)
        a = self.block1(c)
        b = self.block2(c)
        return a+b


x = torch.randn((5, 3, 24, 24))
gen = Encoder_block(64)
summary(gen, input_size=(16, 3, 24, 24))
