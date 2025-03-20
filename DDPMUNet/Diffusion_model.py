import os
import torch
import torch.nn as nn
from eps_net import build_eps_net
from tqdm import tqdm


def img_enc(x):
    return x * 2 - 1


def img_dec(x):
    return torch.clamp((x + 1) / 2, 0, 1)


# x:[batch_size,channels,height,width]
class DDPMUNet(nn.Module):
    def __init__(self, steps, unet_config):
        super(DDPMUNet, self).__init__()
        self.steps = steps
        self.betas = torch.linspace(0.0001, 0.02, steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.eps_net = build_eps_net(
            unet_config, steps
        )  # TODO: predict epsilon according to x_t and t

    def sample_forward_step(self, x: torch.Tensor, t: int, noise=None):
        """
        Samples a forward diffusion step at timestep t.
        Args:
            x (torch.Tensor): The input tensor.
            t (int): The current timestep.
            noise (torch.Tensor, optional): Optional noise tensor. If None, noise will be generated using a standard normal distribution.
        Returns:
            x (torch.Tensor): The resulting tensor after applying the forward diffusion step.
        """

        if noise is None:
            noise = torch.randn_like(x)
        return noise * torch.sqrt(self.alpha_bars[t]) + x * self.alpha_bars[t]

    def _sample_backward_step(self, x_t: torch.Tensor, t: int, simple_variance=False):
        """
        Samples a backward diffusion step at timestep t.
        Args:
            x_t (torch.Tensor): The input tensor at timestep t.
            t (int): The current timestep.
            simple_variance (bool): If True, the variance is simply the beta_t at timestep t. Otherwise, the variance is calculated using the betas and alphas.
        Returns:
            x (torch.Tensor): The resulting tensor after applying the backward diffusion step.
        """

        # noise
        if t == 0:
            noise = 0
        else:
            if simple_variance:
                variance = self.betas[t]
            else:
                variance = (
                    (1 - self.alpha_bars[t - 1])
                    / (1 - self.alpha_bars[t])
                    * self.betas[t]
                )
            noise = torch.randn_like(x_t) * torch.sqrt(variance)

        # mean
        t_tensor = torch.full((x_t.shape[0], 1), t, dtype=torch.long)
        eps = self.eps_net(x_t, t_tensor)
        mean = (
            x_t - (1 - self.alphas[t]) * eps / torch.sqrt(1 - self.alpha_bars[t])
        ) / torch.sqrt(self.alphas[t])

        return mean + noise

    def sample_backward(
        self,
        x: torch.Tensor,
        simple_variance: bool = False,
        random_start: bool = True,
        return_samples_num: int = 20,
    ):
        """
        Samples a backward diffusion step at timestep t. If random_start is True, x should be an arbitrary tensor of the same shape as the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
            simple_variance (bool): If True, the variance is simply the beta_t at timestep t. Otherwise, the variance is calculated using the betas and alphas.
            random_start (bool): If True, the input tensor is ignored and a random tensor is used as the starting point.
            return_samples_num (int): The number-1 of samples to return.
        Returns:
            x (torch.Tensor): The resulting tensor after applying the backward diffusion step.
        """
        if random_start:
            x = torch.randn_like(x)
        lst = [x]
        for t in tqdm(
            reversed(range(self.steps)),
            desc="Sampling backward steps",
            total=self.steps,
            unit="step",
        ):
            x = self._sample_backward_step(x, t, simple_variance)
            if t % (self.steps // return_samples_num) == 0:
                lst.append(x)
        return lst


# check if the model is working
if __name__ == "__main__":
    config = {
        "model_depth": 1000,
        "dataset": "CIFAR10",
        "batch_size": 128,
        "epochs": 32,
        "data_dir": "d:\\Works\\reproducing_diffusion_model\\datasets",
        "model_dir": "d:\\Works\\reproducing_diffusion_model\\models",
        "unet_config": {
            "pic_shape": [3, 32, 32],
            "channels": [10, 20, 40, 80, 160],
            "pe_dim": 256,
            "residual": True,
        },
    }
    model = DDPMUNet(10, config["unet_config"])
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: img_enc(x)),  # Normalize to [-1, 1]
            transforms.Lambda(lambda x: x.float()),
        ]
    )
    ds = torchvision.datasets.CIFAR10(
        root=config["data_dir"], train=True, download=True, transform=transform
    )
    x = ds[0][0].unsqueeze(0)
    print(x.shape)  # torch.Size([1, 3, 32, 32])

    plt.figure()
    plt.subplot(1, 11, 1)
    plt.imshow(x.squeeze().permute(1, 2, 0))
    plt.axis("off")
    for t in range(10):
        x = model.sample_forward_step(x, t)
        plt.subplot(1, 11, t + 2)
        plt.imshow(img_dec(x.detach().cpu().squeeze().permute(1, 2, 0)))
        plt.axis("off")

    plt.show()
    *_, x = model.sample_backward(x, random_start=False)
    plt.imshow(img_dec(x.detach().squeeze().cpu().permute(1, 2, 0)))
    plt.axis("off")
    plt.show()
