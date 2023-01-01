from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms


def getdata():
    trans = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])

    mnist_train = datasets.KMNIST(root='./data', train=True, download=True, transform=trans)

    mnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=trans)

    train_dataloader = DataLoader(mnist_train, batch_size=8, shuffle=True)

    test_dataloader = DataLoader(mnist_test, batch_size=8, shuffle=False)

    return train_dataloader, test_dataloader
