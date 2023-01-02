import torch
import pickle
from PIL import Image


class CifarTrainPair(torch.utils.data.Dataset):
    """Cifar dataloader, output image and target"""

    def __init__(self, image_path, transform=None):
        with open(image_path, 'rb') as f:
            self.images = pickle.load(f)
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None
        img = self.images[index]
        img = Image.fromarray(img)

        img1 = self.transform(img)
        img2 = self.transform(img)

        return img1, img2, 

    def __len__(self):
        return len(self.images)


class CifarDataset(torch.utils.data.Dataset):
    """Cifar dataloader, output image and target"""

    def __init__(self, image_path, target_path, transform=None):
        with open(image_path, 'rb') as f:
            self.images = pickle.load(f)
        with open(target_path, 'rb') as f:
            self.targets = pickle.load(f)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)

class CelebTrainPair(torch.utils.data.Dataset):
    """Celeb dataloader, output image and target"""

    def __init__(self, image_path, transform=None):
        with open(image_path, 'rb') as f:
            self.images = pickle.load(f)
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None
        img = self.images[index]
        img = Image.fromarray(img)

        img1 = self.transform(img)
        img2 = self.transform(img)

        return img1, img2, 

    def __len__(self):
        return len(self.images)
class CelebDataset(torch.utils.data.Dataset):
    """Celeb dataloader, output image and target.
    For now, same as cifardataset loader. """

    def __init__(self, image_path, target_path, transform=None):
        with open(image_path, 'rb') as f:
            self.images = pickle.load(f)
        with open(target_path, 'rb') as f:
            self.targets = pickle.load(f)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)