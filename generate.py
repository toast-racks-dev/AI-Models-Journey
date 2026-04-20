"""
generate.py
-----------
Inference script: load a trained Generator and generate 20 sample images.

Usage:
    python generate.py --checkpoint ./logs/DCGAN_CIFAR10/model_latest.pt
    python generate.py --checkpoint ./logs/DCGAN_CIFAR10/model_latest.pt --output ./my_samples

Output:
    <output_dir>/
    ├── grid.png           4x5 grid of all 20 images (easy visual overview)
    ├── image_00.png
    ├── image_01.png
    ├── ...
    └── image_19.png

Each image is a 32x32 PNG saved at its native resolution.
The grid is saved at native resolution as a composite (4 columns x 5 rows).
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from config import Config
from dcgan import Generator32


NUM_IMAGES = 20    # Fixed: generate exactly 20 images
GRID_NROW  = 4     # Grid layout: 4 columns x 5 rows = 20 images


@torch.no_grad()
def generate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating on: {device}")

    # ---- Load checkpoint -------------------------------------------------- #
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    step = ckpt.get("step", "unknown")
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Checkpoint step:   {step}\n")

    # ---- Build Generator -------------------------------------------------- #
    G = Generator32(z_dim=args.z_dim).to(device)
    G.load_state_dict(ckpt["net_G"])
    G.eval()

    # ---- Sample latent vectors -------------------------------------------- #
    # If a seed is set, the same z always produces the same images.
    # Remove torch.manual_seed to get random samples each run.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Using fixed seed: {args.seed}")

    z = torch.randn(NUM_IMAGES, args.z_dim, device=device)

    # ---- Generate --------------------------------------------------------- #
    fake = G(z).cpu()               # (20, 3, 32, 32) in [-1, 1]
    fake = (fake + 1.0) / 2.0       # Rescale to [0, 1] for saving

    # Resize images to 128x128 so they are big enough to actually see!
    # We use 'nearest' so the pixels stay sharp instead of getting blurry.
    fake = F.interpolate(fake, size=(128, 128), mode='nearest')

    # ---- Save individual images ------------------------------------------- #
    os.makedirs(args.output, exist_ok=True)

    for i, img in enumerate(fake):
        path = os.path.join(args.output, f"image_{i:02d}.png")
        save_image(img, path)

    print(f"Saved {NUM_IMAGES} individual images to: {args.output}/")

    # ---- Save combined grid ----------------------------------------------- #
    # make_grid: arranges (N, C, H, W) into a grid image
    # padding=2: 2-pixel white border between images
    # normalize=False: already in [0,1]
    grid = make_grid(fake, nrow=GRID_NROW, padding=2, normalize=False)
    grid_path = os.path.join(args.output, "grid.png")
    save_image(grid, grid_path)
    print(f"Saved 4x5 image grid to:  {grid_path}")

    print(f"\nDone. All outputs in: {os.path.abspath(args.output)}")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    cfg = Config()

    parser = argparse.ArgumentParser(
        description="Generate 20 images from a trained DCGAN checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./generated",
        help="Directory to save generated images (default: ./generated)",
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=cfg.z_dim,
        help=f"Latent dimension, must match training (default: {cfg.z_dim})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible generation (default: 0, set to -1 for random)",
    )

    args = parser.parse_args()

    # Allow --seed -1 to mean "no fixed seed"
    if args.seed == -1:
        args.seed = None

    generate(args)
