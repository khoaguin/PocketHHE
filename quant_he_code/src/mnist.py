import os

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


from torchvision.datasets import MNIST
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data import random_split
import time

import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant


def square_act(x):
    return torch.mul(x, x)


class SquareAct(nn.Module):
    def __init__(self):
        super(SquareAct, self).__init__()

    def forward(self, x):
        return torch.mul(x, x)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch + 1, result['val_loss'], result['val_acc']))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, test_loader, file_name, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    high_acc = 0.98
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

        if epoch >= 2:
            eval = evaluate(model, test_loader)
            print(str(epoch) + "\t" + str(eval))
            if eval['val_acc'] > high_acc:
                high_acc = eval['val_acc']
                torch.save(model.state_dict(), file_name)
                print("Saved")

    return history


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')


class MNISTModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, stride=(2, 2),
                               padding=0, bias=True)

        self.conv2 = nn.Conv2d(5, 50, 5, stride=(2, 2),
                               padding=0, bias=True)
        self.fc1 = nn.Linear(800, 10, bias=True)


        # self.conv1 = qnn.QuantConv2d(1, 5, 5, stride=(2, 2),
        #                              padding=0, bias=True, weight_bit_width=16, return_quant_tensor=True)
        # self.conv2 = qnn.QuantConv2d(5, 50, 5, stride=(2, 2),
        #                              padding=0, bias=True, weight_bit_width=16)
        #
        # # self.fc1 = qnn.QuantLinear(500, 32, bias=True, weight_bit_width=3, return_quant_tensor=True)
        # self.fc1 = qnn.QuantLinear(800, 10, bias=True, weight_bit_width=16, return_quant_tensor=True)

        self.act1 = SquareAct()
        self.act2 = SquareAct()
        # self.act3 = nn.Sigmoid()

    def forward(self, xb):
        # out = self.quant_inp(xb)
        # out = (((xb*4).int())/4).float() # Scaling to 2bit
        # print(np.min(out.numpy()))
        # print(np.max(out.numpy()))
        # print(out.numpy()[out.numpy() != 0.0])
        out = self.conv1(xb)
        out = out * out
        # out = F.avg_pool2d(out, 3, stride=1, padding=0, divisor_override=1)
        out = self.conv2(out)
        # out = F.avg_pool2d(out, 3, stride=1, padding=1, divisor_override=1)
        out = out.reshape(out.shape[0], -1)
        out = out * out
        out = self.fc1(out)
        # out = out * out
        # out = self.fc2(out)
        # out = self.act3(out)
        return out


def main():
    # Dataset
    dataset = MNIST(root='data/', download=True, transform=transforms.Compose([
        ToTensor(),
        lambda x: (x*4).int(),
        lambda x: x.float()/4,
    ]))
    test_dataset = MNIST(root='data/', train=False, transform=transforms.Compose([
        ToTensor(),
        lambda x: (x*4).int(),
        lambda x: x.float()/4,
    ]))

    # images, labels = next(iter(dataset))
    # print(images[0])

    # Preparing data for training
    torch.manual_seed(43)
    val_size = 5000
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = 128

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size * 2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size * 2, pin_memory=True)

    # Get GPU or CPU
    device = get_default_device()

    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    model = to_device(MNISTModel(), device)

    history = [evaluate(model, val_loader)]
    savefile_name = "hcnn_mnist_plain"
    history += fit(150, 0.001, model, train_loader, val_loader, test_loader, str(savefile_name + ".pth"), torch.optim.Adam)

    # pt for brevitas
    # pth for regular pytorch
    # plot_losses(history)
    # plot_accuracies(history)

    print(evaluate(model, test_loader))

    # os.system("python export_weights_py.py " + str(savefile_name) + ".pt " + str(savefile_name + "_data.py"))

    # mnist_hcnn_in_2: 98.69% (with quantisation of the input)

    # CryptoNets 3b: 98.84
    # CryptoNets 2b: 98.77

    # Own MNIST 3b: 98.91
    # Own MNIST 2b: 98.79

    # Completly HCNN; only stride=(1,1)
    # HCNN_mnist_3 : 98.57
    # HCNN_mnist_2 : 98.00

    # Completly HCNN; only stride=(2,2)
    # HCNN_mnist_2 : 98.46
    # HCNN_mnist_3 : 98.56
    # HCNN_mnist_4 : 98.65
    # HCNN_mnist_6: 98.53
    # HCNN_mnist_8: 98.53
    # HCNN_mnist_16: 98.51
    # HCNN_mnist_32: 98.39

    # HCNN_mnist_plain: 98.60



if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'--- {time.time() - start_time:.3f} seconds --- \n --- {(time.time() - start_time) / 60:.3f} minutes ---')
