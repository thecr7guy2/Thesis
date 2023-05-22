import torch
from collections import OrderedDict

alpha = 0.8
alpha = float(alpha)
net_PSNR_path = "bestgen.pth.tar"
net_ESRGAN_path = "gen160.pth.tar"
net_interp_path = 'interp_{:02d}.pth.tar'.format(int(alpha * 10))

net_PSNR = torch.load(net_PSNR_path)
net_PSNR = net_PSNR["state_dict"]
net_ESRGAN = torch.load(net_ESRGAN_path)
net_ESRGAN = net_ESRGAN["state_dict"]
net_interp = OrderedDict()

print('Interpolating with alpha = ', alpha)
for k, v_PSNR in net_PSNR.items():
    v_ESRGAN = net_ESRGAN[k]
    net_interp[k] = ((1 - alpha) * v_PSNR) + (alpha * v_ESRGAN)

torch.save(net_interp, net_interp_path)
