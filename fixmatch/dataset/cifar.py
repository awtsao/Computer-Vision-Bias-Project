import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import pickle
import torch

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFARDataset(Dataset):
    """Cifar dataloader, output image and target"""

    def __init__(self, image_path, class_label_path, transform=None):
        with open(image_path, 'rb') as f:
            self.images = pickle.load(f)
        with open(class_label_path, 'rb') as f:
            self.class_label = pickle.load(f)
        self.transform = transform

        self.class_label = torch.tensor(self.class_label)

    def __getitem__(self, index):
        img, target = self.images[index], self.class_label[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.class_label)


class CIFARDatasetSSL(CIFARDataset):
    def __init__(self, image_path, class_label_path, transform=None, target_transform=None, indexs=None):
        super().__init__(image_path, class_label_path, transform)
        self.target_transform = target_transform

        if indexs is not None:
            self.images = self.images[indexs]
            self.class_label = self.class_label[indexs]

    def __getitem__(self, index):
        img, class_lbl = self.images[index], self.class_label[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            class_lbl = self.target_transform(class_lbl)

        return img, class_lbl

    def __len__(self):
        return len(self.class_label)


class CIFARDatasetWithDomain(Dataset):
    """Cifar dataloader, output image, class target and domain for this sample"""

    def __init__(self, image_path, class_label_path, domain_label_path, transform=None):
        with open(image_path, 'rb') as f:
            self.images = pickle.load(f)
        with open(class_label_path, 'rb') as f:
            self.class_label = pickle.load(f)
        with open(domain_label_path, 'rb') as f:
            self.domain_label = pickle.load(f)
        self.transform = transform

        self.class_label = torch.tensor(self.class_label)
        self.domain_label = torch.tensor(self.domain_label)

    def __getitem__(self, index):
        img, class_label, domain_label = \
            self.images[index], self.class_label[index], self.domain_label[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, class_label, domain_label

    def __len__(self):
        return len(self.class_label)


class CIFARDatasetWithDomainSSL(CIFARDatasetWithDomain):
    def __init__(self, image_path, class_label_path, domain_label_path, transform=None, target_transform=None, indexs=None):
        super().__init__(image_path, class_label_path, domain_label_path, transform)
        self.target_transform = target_transform

        if indexs is not None:
            self.images = self.images[indexs]
            self.class_label = self.class_label[indexs]
            self.domain_label = self.domain_label[indexs]

    def __getitem__(self, index):
        img, class_lbl, domain_lbl = self.images[index], self.class_label[index], self.domain_label[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            class_lbl = self.target_transform(class_lbl)
            domain_lbl = self.target_transform(domain_lbl)

        return img, class_lbl, domain_lbl

    def __len__(self):
        return len(self.class_label)


def get_cifar10s(args, image_path, class_label_path):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = CIFARDataset(
        image_path, class_label_path)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.class_label)

    train_labeled_dataset = CIFARDatasetSSL(
        image_path, class_label_path, indexs=train_labeled_idxs,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFARDatasetSSL(
        image_path, class_label_path, indexs=train_unlabeled_idxs,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = CIFARDataset(
        image_path, class_label_path, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar10s_domain(args, image_path, class_label_path, domain_label_path):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = CIFARDatasetWithDomain(
        image_path, class_label_path, domain_label_path)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.class_label)

    train_labeled_dataset = CIFARDatasetWithDomainSSL(
        image_path, class_label_path, domain_label_path, indexs=train_labeled_idxs,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFARDatasetWithDomainSSL(
        image_path, class_label_path, domain_label_path, indexs=train_unlabeled_idxs,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = CIFARDatasetWithDomain(
        image_path, class_label_path, domain_label_path, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'cifar10s': get_cifar10s,
                   'cifar10s_domain': get_cifar10s_domain,
                   'cifar10s_discriminative': get_cifar10s_domain}
