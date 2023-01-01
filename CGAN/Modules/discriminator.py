import torch.nn as nn


class discriminator(nn.Module):
    def __init__(self, im_dim, hidden_dim):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            self.dis_block(im_dim, hidden_dim, 4, 2, final_layer=False),
            self.dis_block(hidden_dim, hidden_dim * 2, 4, 2, final_layer=False),
            self.dis_block(hidden_dim * 2, 1, 4, 2, final_layer=True),
        )

    @staticmethod
    def dis_block(input_channels, output_channels, kernel_size, stride, final_layer=False):

        if final_layer == False:
            block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        else:
            block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )
        return block

    def forward(self, image):
        x = self.dis(image)
        x = x.view(len(x), -1)
        return x
