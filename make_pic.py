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
    entropy = -(frame * torch.log2(frame + 1e-20) + (1 - frame) * (torch.log2(1.0 - frame + 1e-20)))
    return torch.sum(entropy)


def spacial_entropy(frame, size=3):
    num_buckets = 2**(size**2)
    buckets = torch.zeros(num_buckets)
    for bucket in range(num_buckets):
        _, x, y = frame.shape
        base = torch.zeros((1, x-size+1, y-size+1))
        for i in range(size*size):
            start_x = i // size
            end_x = None if start_x == size - 1 else (start_x - (size - 1))
            start_y = i % size
            end_y = None if start_y == size - 1 else (start_y - (size - 1))
            cut_frame = frame[:, start_x:end_x, start_y:end_y]
            neg = bool(bucket & (1 << i))
            base += torch.log(cut_frame if neg else (1.0 - cut_frame)) 
        base = torch.exp(base)
        buckets[bucket] = torch.sum(base)
    normed_buckets = buckets / torch.sum(buckets)
    # print(buckets)
    entropy = torch.sum(normed_buckets * torch.log(normed_buckets + 1e-20))
    # print(entropy)
    return entropy


def model_loss(model, ic, length=5):
    frames = [ic]
    for i in range(length):
        frames.append(model(frames[-1]))
        assert frames[0].shape == frames[1].shape, f"{new.shape} {last.shape}"
    frames.pop(0)

    loss = 0
    for frame in frames:
        # loss += binary_entropy(frame)
        loss += spacial_entropy(frame)
        # loss += torch.sum(-frame * (frame - 1.0))

    # for i in range(len(frames)):
    #     sub = 1
    #     while (i - sub >= 0):
    #         loss += torch.sum(1.0 - (frames[i] - frames[i - sub]) ** 2.0)
    #         sub *= 2

    return loss


args = parse_args()

model = NeuralConway()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
try:
    for i in range(50):
        ic = torch.randint(0, 2, (1, args.size, args.size)) + 0.
        loss = model_loss(model, ic)
        print(i, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
except KeyboardInterrupt:
    pass

make_random_gif(model, args.outfile, args.size, args.frames)
