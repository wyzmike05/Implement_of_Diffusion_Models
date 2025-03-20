import torch
import os
import json

from prepare_datasets import get_dataloader
from Diffusion_model import DDPMUNet
from train_model import train_model


config = {
    "model_depth": 1000,
    "dataset": "CelebA-HQ",  # "MNIST" or "CIFAR10" or "CelebA-HQ"
    "batch_size": 4,  # 64 in paper
    "epochs": 32,  # 1142 in paper
    "data_dir": os.path.join(os.path.dirname(__file__), "datasets"),
    "model_dir": os.path.join(os.path.dirname(__file__), "models"),
}


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
dataloader, pic_shape = get_dataloader(
    config["data_dir"], config["dataset"], config["batch_size"]
)

unet_config = {
    "pic_shape": pic_shape,
    "channels": [10, 20, 40, 80, 160],
    "pe_dim": 256,
    "residual": True,
}

config["unet_config"] = unet_config

# Load the model
os.makedirs(config["model_dir"], exist_ok=True)

with open(os.path.join(config["model_dir"], "config.json"), "w") as f:
    json.dump(config, f)

model = DDPMUNet(config["model_depth"], unet_config)


LOAD_MODEL = False
CHECKPOINT = 6  # -1 for no checkpoint
if LOAD_MODEL:
    try:
        model.load_state_dict(
            torch.load(os.path.join(config["model_dir"], f"model.pth"))
        )
        print("Model loaded successfully.")
    except FileNotFoundError:
        raise FileNotFoundError("Model not found. Please train a new model instead.")
    except RuntimeError:
        raise RuntimeError(
            "Model with improper parameters. Please train a new model instead."
        )

elif CHECKPOINT >= 0:
    try:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    config["model_dir"], f"checkpoint_epoch_{CHECKPOINT:04d}.pth"
                )
            )
        )
        print(f"checkpoint_epoch_{CHECKPOINT:04d} loaded successfully.")
        assert config["epochs"] >= CHECKPOINT
        train_model(
            model,
            dataloader,
            device,
            config["model_dir"],
            config["epochs"],
            CHECKPOINT,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "Checkpoint not found. Please train a new model instead."
        )
    except RuntimeError:
        raise RuntimeError(
            "Checkpoint with improper parameters. Please train a new model instead."
        )
else:
    train_model(model, dataloader, device, config["model_dir"], config["epochs"])
