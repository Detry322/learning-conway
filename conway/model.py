import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralConway(nn.Module):
    def __init__(self):
        super(NeuralConway, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, padding=2, padding_mode='circular')
        self.conv2 = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, *x.shape[1:])
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        x = x.reshape(x.shape[0], *x.shape[2:])
        return x

    def load_conway_weights(self, multiplier=100):
        self.load_state_dict({
            'conv1.weight': torch.tensor([
                [[[-1.0, -1.0, -1.0],
                  [-1.0, -1.0, -1.0],
                  [-1.0, -1.0, -1.0]]],
                [[[-1.0, -1.0, -1.0],
                  [-1.0, -1.0, -1.0],
                  [-1.0, -1.0, -1.0]]],
                [[[ 1.0,  1.0,  1.0],
                  [ 1.0,  0.0,  1.0],
                  [ 1.0,  1.0,  1.0]]]]) * multiplier,
            'conv1.bias': torch.tensor([ 2.5,  2.5, -3.5]) * multiplier,
            'conv2.weight': torch.tensor([[[[-1.0]], [[-1.0]], [[-2.0]]]]) * multiplier,
            'conv2.bias': torch.tensor([0.5]) * multiplier
        })
