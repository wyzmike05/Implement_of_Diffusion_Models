import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from Diffusion_model import DDPMUNet


def train_model(
    model: DDPMUNet,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str,
    epochs: int = 10,
    checkpoint: int = -1,
):

    model.to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)

    steps = model.steps
    with open(os.path.join(save_path, f"log.txt"), "w") as f:
        if checkpoint >= 0:
            start_epoch = checkpoint
        else:
            start_epoch = 0
        for epoch in range(start_epoch, epochs):
            running_loss = 0
            print(f"Epoch {epoch+1}/{epochs}:")
            for x_0, _ in tqdm(dataloader):
                batch_size = x_0.shape[0]
                x_0 = x_0.to(device)
                t = torch.randint(0, steps, (batch_size,)).to(device)
                eps = torch.randn_like(x_0).to(device)
                alpha_bars_t = model.alpha_bars[t].view(batch_size, 1, 1, 1)
                eps_theta = model.eps_net(
                    torch.sqrt(alpha_bars_t) * x_0 + torch.sqrt(1 - alpha_bars_t) * eps,
                    t.reshape(batch_size, 1),
                )
                loss = criterion(eps_theta, eps)
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            torch.save(
                model.state_dict(),
                os.path.join(save_path, f"checkpoint_epoch_{epoch+1:04d}.pth"),
            )

            f.write(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}\n")
    torch.save(
        model.state_dict(),
        os.path.join(save_path, f"model.pth"),
    )
