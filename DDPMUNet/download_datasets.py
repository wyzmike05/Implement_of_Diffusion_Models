import torchvision
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")

ds = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True)
ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True)

from datasets import load_dataset

ds = load_dataset("korexyz/celeba-hq-256x256")
ds.save_to_disk(DATA_DIR + r"\CelebA-HQ")
