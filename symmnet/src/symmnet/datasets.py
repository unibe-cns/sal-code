#!/usr/bin/env python3

import torch
from torchvision import datasets, transforms


def cifar10(batch_size, path="../datasets"):
    n_channels = 3
    shape = (32, 32)
    # Data augmentation and normalization for training
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49, 0.48, 0.44), (0.20, 0.20, 0.20)),
        ]
    )

    # no augmentation for testing
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.49, 0.48, 0.44), (0.2, 0.2, 0.2)),
        ]
    )

    train_set = datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_set = datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, n_channels, shape


def mnist(batch_size, path="../datasets"):
    n_channels = 1
    shape = (28, 28)
    # Data augmentation and normalization for training
    transform_train = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.13,), (0.30,)),
        ]
    )

    # no augmentation for testing
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.13,), (0.30,)),
        ]
    )

    train_set = datasets.MNIST(
        root=path, train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_set = datasets.MNIST(
        root=path, train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, n_channels, shape


def fmnist(batch_size, path="../datasets"):
    n_channels = 1
    shape = (28, 28)
    # Data augmentation and normalization for training
    transform_train = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.29,), (0.32,)),
        ]
    )

    # no augmentation for testing
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.29,), (0.32,)),
        ]
    )

    train_set = datasets.FashionMNIST(
        root=path, train=True, download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_set = datasets.FashionMNIST(
        root=path, train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, n_channels, shape


def imagenette(batch_size, path="../datasets"):
    n_channels = 3
    shape = (160, 160)
    # Data augmentation and normalization for training
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(160, padding=20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # no augmentation for testing
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set = datasets.Imagenette(
        root=path, split="train", download=True, size="160px", transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_set = datasets.Imagenette(
        root=path, split="val", download=True, size="160px", transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=100, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, n_channels, shape


def svhn(batch_size, path="../datasets"):
    n_channels = 3
    shape = (32, 32)
    # Data augmentation and normalization for training
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.44, 0.44, 0.47), (0.12, 0.12, 0.11)),
        ]
    )

    # no augmentation for testing
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.44, 0.44, 0.47), (0.12, 0.12, 0.11)),
        ]
    )

    train_set = datasets.SVHN(
        root=path, split="train", download=True, transform=transform_train
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_set = datasets.SVHN(
        root=path, split="test", download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, n_channels, shape
