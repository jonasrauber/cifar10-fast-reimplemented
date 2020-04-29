#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


def ConvBN(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = ConvBN(c, c)
        self.conv2 = ConvBN(c, c)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return x


class Multiply(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, x):
        return x * self.weight


def net():
    return nn.Sequential(
        ConvBN(3, 64),
        *(ConvBN(64, 128), nn.MaxPool2d(2), Residual(128)),
        *(ConvBN(128, 256), nn.MaxPool2d(2)),
        *(ConvBN(256, 512), nn.MaxPool2d(2), Residual(512)),
        nn.MaxPool2d(4),
        nn.Flatten(),
        nn.Linear(512, 10, bias=False),
        Multiply(0.125),
    )


class Crop:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, x, x0, y0):
        return x[..., y0 : y0 + self.h, x0 : x0 + self.w]

    def options(self, shape):
        *_, H, W = shape
        return [
            {"x0": x0, "y0": y0}
            for x0 in range(W + 1 - self.w)
            for y0 in range(H + 1 - self.h)
        ]

    def output_shape(self, shape):
        *_, H, W = shape
        return (*_, self.h, self.w)


class FlipLR:
    def __call__(self, x, choice):
        return x[..., ::-1].copy() if choice else x

    def options(self, shape):
        return [{"choice": b} for b in [True, False]]


class Cutout:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, x, x0, y0):
        x[..., y0 : y0 + self.h, x0 : x0 + self.w] = 0.0
        return x

    def options(self, shape):
        *_, H, W = shape
        return [
            {"x0": x0, "y0": y0}
            for x0 in range(W + 1 - self.w)
            for y0 in range(H + 1 - self.h)
        ]


class RandomAugmentation(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, labels = self.dataset[index]
        data = data.copy()
        for choices, f in zip(self.choices, self.transforms):
            data = f(data, **choices[index])
        return data, labels

    def sample_transformations(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            self.choices.append(np.random.choice(t.options(x_shape), N))
            x_shape = t.output_shape(x_shape) if hasattr(t, "output_shape") else x_shape


def normalize(x, mean, std):
    x = np.array(x, np.float32)
    x -= mean
    x *= 1.0 / std
    return x


def train(args, model, device, train_loader, optimizer, scheduler, epoch):
    all_losses = []
    all_correct = []

    model.train()
    for images, labels in train_loader:
        images = images.to(device).half()
        labels = labels.to(device).long()

        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="none")
        correct = logits.argmax(dim=1) == labels

        optimizer.zero_grad()
        loss.sum().backward()

        optimizer.step()
        scheduler.step()

        all_losses.append(loss.detach())
        all_correct.append(correct.detach())

    return {
        "loss": torch.cat(all_losses).cpu().numpy().astype(np.float64).mean(),
        "accuracy": torch.cat(all_correct).cpu().numpy().astype(np.float64).mean(),
    }


def test(model, device, test_loader):
    all_losses = []
    all_correct = []

    model.eval()
    for images, labels in test_loader:
        images = images.to(device).half()
        labels = labels.to(device).long()

        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction="none")
        correct = logits.argmax(dim=1) == labels

        all_losses.append(loss.detach())
        all_correct.append(correct.detach())

    return {
        "loss": torch.cat(all_losses).cpu().numpy().astype(np.float64).mean(),
        "accuracy": torch.cat(all_correct).cpu().numpy().astype(np.float64).mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512, metavar="N")
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N')
    parser.add_argument("--epochs", type=int, default=24, metavar="N")
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR')
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument("--data", default="./data/")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    cifar_mean = np.array([125.31, 122.95, 113.87], dtype=np.float32)
    cifar_std = np.array([62.99, 62.09, 66.70], dtype=np.float32)

    train_set = CIFAR10(root=args.data, train=True, download=True)
    data = train_set.data
    data = np.pad(data, [(0, 0), (4, 4), (4, 4), (0, 0)], mode="reflect")
    data = normalize(data, mean=cifar_mean, std=cifar_std)
    data = data.transpose([0, 3, 1, 2])
    train_set = list(zip(data, train_set.targets))
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]
    train_set = RandomAugmentation(train_set, train_transforms)

    test_set = CIFAR10(root=args.data, train=False, download=False)
    data = test_set.data
    data = normalize(data, mean=cifar_mean, std=cifar_std)
    data = data.transpose([0, 3, 1, 2])
    test_set = list(zip(data, test_set.targets))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    model = net().to(device).half()

    optimizer = SGD(
        model.parameters(),
        lr=args.lr / args.batch_size,
        momentum=0.9,
        weight_decay=5e-4 * args.batch_size,
        nesterov=True,
    )

    def piecewise_linear(step):
        epoch = (step + 1) / len(train_loader)
        lr = np.interp([epoch], [0, 5, args.epochs], [0, 0.4, 0])[0]
        return lr

    scheduler = LambdaLR(optimizer, piecewise_linear, last_epoch=-1)

    for epoch in range(1, args.epochs + 1):
        train_set.sample_transformations()
        train_summary = train(
            args, model, device, train_loader, optimizer, scheduler, epoch
        )
        test_summary = test(model, device, test_loader)
        print(
            f"{epoch:3d}. epoch:"
            f" train accuracy = {train_summary['accuracy']:.4f} ({train_summary['loss']:.2e})"
            f" test accuracy = {test_summary['accuracy']:.4f} ({test_summary['loss']:.2e})"
        )

    if args.save_model:
        torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
