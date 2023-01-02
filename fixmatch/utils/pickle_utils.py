import pickle
import numpy as np
import torchvision.transforms as transforms


def getNormalize(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    mean = tuple(np.mean(data / 255., axis=(0, 1, 2)))
    std = tuple(np.std(data / 255., axis=(0, 1, 2)))
    normalize = transforms.Normalize(mean=mean, std=std)
    return normalize


def save_pkl(pkl_data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(pkl_data, f)


def write_info(filename, info):
    with open(filename, 'w') as f:
        f.write(info)
