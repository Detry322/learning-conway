import torch
import torchvision.transforms as transforms
import PIL

def make_random_gif(model, outfile, num_frames=500, size=100):
    start = torch.randint(0, 2, (1, size, size)) + 0.
    frames = [start]
    for i in range(num_frames):
        frames.append(model(frames[-1]))

    images = list(map(
        lambda im: im.resize((im.size[0] * 4, im.size[1] * 4), resample=PIL.Image.NEAREST),
        map(
            transforms.ToPILImage(mode='L'),
            map(
                lambda frame: frame.reshape(frame.shape[1:]),
                frames
            )
        )
    ))
    images[0].save(outfile, save_all=True, append_images=images[1:])


