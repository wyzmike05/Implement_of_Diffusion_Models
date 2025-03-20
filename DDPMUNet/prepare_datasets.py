import torch
import torchvision
from torchvision import transforms
import os
from datasets import load_from_disk


def img_enc(x):
    return x * 2 - 1


def img_dec(x):
    return torch.clamp((x + 1) / 2, 0, 1)


def get_dataloader(data_dir: str, dataset: str, batch_size: int):
    """
    Returns a DataLoader object for the specified dataset.

    Args:
        data_dir (str): The directory where the dataset is stored.
        dataset (str): The name of the dataset.
        batch_size (int): The batch size.
    Returns:
        dataloader (torch.utils.data.DataLoader): The DataLoader
        object for the specified dataset.
    """
    assert dataset in ["MNIST", "CIFAR10", "CelebA-HQ"], "Dataset not supported."
    if dataset == "MNIST":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: img_enc(x)),
                transforms.Lambda(lambda x: x.float()),
            ]
        )
        ds = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True
        )
        shape = (1, 28, 28)
    elif dataset == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: img_enc(x)),
                transforms.Lambda(lambda x: x.float()),
            ]
        )
        ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True
        )
        shape = (3, 32, 32)
    elif dataset == "CelebA-HQ":
        ds = load_from_disk(os.path.join(data_dir, "CelebA-HQ", "train"))

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: img_enc(x)),
                transforms.Lambda(lambda x: x.float()),
            ]
        )

        class CelebA_HQ:
            def __init__(self, ds, transform):
                self.ds = ds
                self.transform = transform

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, idx):
                img, label = self.ds[idx]["image"], self.ds[idx]["label"]
                img = self.transform(img)
                res = [img, label]
                return res

        ds = CelebA_HQ(ds, transform)

        dataloader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True
        )
        shape = (3, 256, 256)

    return dataloader, shape


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dl, pic_shape = get_dataloader(
        os.path.join(os.path.dirname(__file__), "datasets"), "CelebA-HQ", 4
    )

    for x_0, _ in dl:
        sample = x_0[0]
        plt.imshow(img_dec(sample).permute(1, 2, 0))
        plt.show()
        break
