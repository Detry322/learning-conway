import argparse
import torch
from conway.model import NeuralConway
from conway.display import make_random_gif


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("outfile")
    parser.add_argument("size", type=int)
    parser.add_argument("frames", type=int)
    return parser.parse_args()


def binary_entropy(frame):
    entropy = -(frame * torch.log2(frame) + (1 - frame) * (torch.log2(1.0 - frame)))
    return torch.sum(entropy)


def weird_quad(frame):
    return (frame - 1.0) * frame * frame * (frame + 1.0) + 0.25


def model_loss(model, ic, length=25):
    frames = [ic]
    for i in range(length):
        frames.append(model(frames[-1]))
        assert frames[0].shape == frames[1].shape, f"{new.shape} {last.shape}"

    loss = 0
    for frame in frames:
        loss += torch.sum(-frame * (frame - 1.0))

    for i in range(len(frames)):
        sub = 1
        while (i - sub >= 0):
            loss += torch.sum(1.0 - (frames[i] - frames[i - sub]) ** 2.0)
            sub *= 2

    return loss


args = parse_args()

model = NeuralConway()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
for i in range(50):
    ic = torch.randint(0, 2, (1, args.size, args.size)) + 0.
    loss = model_loss(model, ic)
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

make_random_gif(model, args.outfile, args.size, args.frames)
