"""
evaluate.py
-----------
Standalone evaluation script: load a trained checkpoint and compute FID + IS.

Usage:
    python evaluate.py --checkpoint ./logs/DCGAN_CIFAR10/model_latest.pt
    python evaluate.py --checkpoint ./logs/DCGAN_CIFAR10/checkpoints/ckpt_step_0100000.pt

What it does:
  1. Loads the Generator from the checkpoint.
  2. Generates `num_images` (default 50,000) images.
  3. Computes Inception Score (IS) and Frechet Inception Distance (FID).
  4. Prints results.

No external dependencies needed — uses our custom metrics.py
(torchvision InceptionV3 + scipy, both pre-installed on Kaggle).
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import Config
from dcgan import Generator32
from metrics import compute_real_stats, compute_fid_and_is


@torch.no_grad()
def generate_images(
    G: torch.nn.Module,
    device: torch.device,
    z_dim: int,
    num_images: int,
    batch_size: int,
) -> torch.Tensor:
    """
    Generate num_images images from Generator G.

    Returns:
        Tensor (num_images, 3, 32, 32) with pixel values in [0, 1].
    """
    G.eval()
    images = []

    generated = 0
    while generated < num_images:
        n = min(batch_size, num_images - generated)
        z = torch.randn(n, z_dim, device=device)
        imgs = G(z).cpu()          # [-1, 1]
        imgs = (imgs + 1.0) / 2.0  # [0, 1]
        images.append(imgs)
        generated += n

        if generated % 10_000 == 0 or generated == num_images:
            print(f"  Generated {generated:>6}/{num_images} images...")

    return torch.cat(images, dim=0)


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # ---- Load checkpoint -------------------------------------------------- #
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    step = ckpt.get("step", "unknown")
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Checkpoint step:   {step}")

    # ---- Build Generator -------------------------------------------------- #
    z_dim = args.z_dim
    G = Generator32(z_dim=z_dim).to(device)
    G.load_state_dict(ckpt["net_G"])
    G.eval()
    print(f"Generator loaded. z_dim={z_dim}\n")

    # ---- Ensure FID cache exists ------------------------------------------ #
    if os.path.exists(args.fid_cache):
        print(f"FID cache found: {args.fid_cache}")
        data = np.load(args.fid_cache)
        real_mu, real_sigma = data["mu"], data["sigma"]
    else:
        print(f"FID cache not found at: {args.fid_cache}")
        print("Computing CIFAR-10 Inception stats (one-time cost)...\n")

        ds = datasets.CIFAR10(
            root="./data", train=True, download=True,
            transform=transforms.ToTensor(),
        )
        dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)
        all_imgs = torch.cat([b for b, _ in dl], dim=0)
        real_mu, real_sigma = compute_real_stats(all_imgs, device, batch_size=64)

        os.makedirs(os.path.dirname(args.fid_cache), exist_ok=True)
        np.savez(args.fid_cache, mu=real_mu, sigma=real_sigma)
        print(f"FID cache saved to: {args.fid_cache}\n")

    # ---- Generate images -------------------------------------------------- #
    print(f"Generating {args.num_images:,} images...")
    imgs = generate_images(G, device, z_dim, args.num_images, args.batch_size)
    print(f"Generation complete. Image tensor shape: {imgs.shape}\n")

    # ---- Compute FID + IS ------------------------------------------------- #
    print("Computing Inception Score and FID (this takes a few minutes)...")
    IS_mean, IS_std, FID = compute_fid_and_is(
        imgs, real_mu, real_sigma, device, batch_size=64
    )

    # ---- Results ---------------------------------------------------------- #
    print("\n" + "=" * 50)
    print(f"  Checkpoint step:  {step}")
    print(f"  Num images:       {args.num_images:,}")
    print(f"  Inception Score:  {IS_mean:.3f} ± {IS_std:.5f}")
    print(f"  FID:              {FID:.3f}")
    print("=" * 50)
    print("\nExpected ballpark for DCGAN on CIFAR-10:")
    print("  FID: ~30–50   (lower is better)")
    print("  IS:  ~6.5–7.5 (higher is better)")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    cfg = Config()

    parser = argparse.ArgumentParser(description="Evaluate DCGAN with FID and IS")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--z_dim",
        type=int,
        default=cfg.z_dim,
        help=f"Latent dimension (default: {cfg.z_dim})",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=cfg.num_images,
        help=f"Number of images to generate for FID/IS (default: {cfg.num_images})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=cfg.batch_size,
        help=f"Batch size for generation (default: {cfg.batch_size})",
    )
    parser.add_argument(
        "--fid_cache",
        type=str,
        default=cfg.fid_cache,
        help=f"Path to FID stats cache .npz (default: {cfg.fid_cache})",
    )

    args = parser.parse_args()
    evaluate(args)
