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

    def load_conway_weights(self):
        self.load_state_dict({
            'conv1.weight': torch.tensor([
                [[[-100.0, -100.0, -100.0],
                  [-100.0, -100.0, -100.0],
                  [-100.0, -100.0, -100.0]]],
                [[[-100.0, -100.0, -100.0],
                  [-100.0, -100.0, -100.0],
                  [-100.0, -100.0, -100.0]]],
                [[[ 100.0,  100.0,  100.0],
                  [ 100.0,    0.0,  100.0],
                  [ 100.0,  100.0,  100.0]]]]),
            'conv1.bias': torch.tensor([ 250.0,  250.0, -350.0]),
            'conv2.weight': torch.tensor([[[[-100.0]], [[-100.0]], [[-200.0]]]]),
            'conv2.bias': torch.tensor([50.0])
        })
