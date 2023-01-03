import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt


def getdata():
    trans = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])

    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=trans)

    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=trans)

    train_dataloader = DataLoader(mnist_train, batch_size=8, shuffle=True)

    test_dataloader = DataLoader(mnist_test, batch_size=8, shuffle=False)

    return train_dataloader, test_dataloader

# train,test = getdata()
#
# images,labels =next(iter(train))
# rows = 2
# columns = 2
# fig=plt.figure()
# for i in range(4):
#    fig.add_subplot(rows, columns, i+1)
#    img = images[i].numpy().transpose((1, 2, 0))
#    plt.imshow(img)
# plt.show()
