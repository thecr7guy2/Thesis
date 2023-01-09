import torch
import torch.nn as nn


class discriminator(nn.Module):
    """Some Information about discriminator"""

    def __init__(self, im_dim, hidden_dim):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            self.dis_block(im_dim, hidden_dim, 4, 2, 1, final_layer=False),
            self.dis_block(hidden_dim, hidden_dim * 2, 4, 2, 1, final_layer=False),
            self.dis_block(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, final_layer=False),
            self.dis_block(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, final_layer=False),
            self.dis_block(hidden_dim * 8, 1, 4, 2, 0, final_layer=True),
        )

    @staticmethod
    def dis_block(input_channels, output_channels, kernel_size, stride, padding, final_layer=False):

        if final_layer == False:
            block = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
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


def dis_loss(gen, disc, criterion, real_im, noise_dim, device):
    noise_vec = torch.randn(len(real_im), noise_dim, device=device)

    # pass the noise as an input to the generator to generate fake images

    fake_images = gen(noise_vec)

    # Now use the discriminator and get the predictions of the generated fake images

    pred_fakes = disc(fake_images.detach())

    # Create the ground truth vector

    ground_fakes = torch.zeros_like(pred_fakes)

    # Ue the cost function to compute the loss caused by fake images

    fake_loss = criterion(pred_fakes, ground_fakes)

    # repeat similar process for the real images

    pred_real = disc(real_im)

    # Create the ground truth vector

    ground_real = torch.ones_like(pred_real)

    real_loss = criterion(pred_real, ground_real)

    loss = (real_loss + fake_loss) / 2

    return loss


# batch_size = 2
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# dis = discriminator(1, 16).to(device)
# images = torch.normal(0, 1, size=(batch_size, 1, 28, 28)).to(device)
# print(images.shape)
# # images = images.view(batch_size, -1).to(device)
# # print(images.shape)
# print("########################")
# pred = dis(images)
# print(pred.shape)
# print(pred)
