import os
import torch
import cv2
import json
import matplotlib.pyplot as plt

from Diffusion_model import DDPMUNet

# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True


with open(os.path.join(os.path.dirname(__file__), "models", "config.json"), "r") as f:
    config = json.load(f)
unet_config = config["unet_config"]

model = DDPMUNet(config["model_depth"], unet_config)

test_model = "model.pth"  # f"model.pth"


try:
    print(
        "Trying to load model from",
        os.path.join(config["model_dir"], test_model),
    )
    model.load_state_dict(torch.load(os.path.join(config["model_dir"], test_model)))
    print("Model/Checkpoint loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(
        "Model/Checkpoint not found. Please train a new model instead."
    )
except RuntimeError:
    raise RuntimeError(
        "Model/Checkpoint with improper parameters. Please train a new model instead."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# k samples to generate
k = 5


def img_dec(x):
    return torch.clamp((x + 1) / 2, 0, 1)


model.to(device)
model.eval()
with torch.no_grad():
    x_shape = unet_config["pic_shape"]
    x_shape = [k] + x_shape
    sample = torch.zeros(x_shape).to(device)
    x_sample = model.sample_backward(sample, random_start=True)

    plt.figure(figsize=(100, 40))
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    for idx in range(k):
        for t in range(len(x_sample)):
            plt.subplot(k, len(x_sample), idx * len(x_sample) + t + 1)
            pic = x_sample[t][idx].cpu().permute(1, 2, 0)
            pic = img_dec(pic)
            if config["dataset"] == "MNIST":
                plt.imshow(pic, cmap="gray")
            else:
                plt.imshow(pic)
            plt.axis("off")
    plt.savefig(os.path.join(config["model_dir"], "samples.png"))
    plt.close()
