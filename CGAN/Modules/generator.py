import torch
import torch.nn as nn


class generator(nn.Module):
    def __int__(self, input_dim, hidden_dim, im_dim):
        super(generator, self).__int__()
        self.generator = nn.Sequential(
            self.generator_block(input_dim, hidden_dim * 4, 3, 2, 0, last_layer=False),
            self.generator_block(hidden_dim * 4, hidden_dim * 2, 4, 1, 0, last_layer=False),
            self.generator_block(hidden_dim * 2, hidden_dim, 3, 2, 0, last_layer=False),
            self.generator_block(hidden_dim, im_dim, 4, 2, 0, last_layer=True)
        )

    @staticmethod
    def generator_block(in_channels, out_channels, kernel_size, stride, padding, last_layer=False):
        if last_layer == False:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.Tanh()
            )

        return block

    def forward(self, noise):
        noise = noise.view(noise.shape[0], noise.shape[1], 1, 1)
        gen_image = self.generator(noise)
        return gen_image
