"""
config.py
---------
All hyperparameters and paths for the DCGAN CIFAR-10 experiment.
Edit this file before running train.py.

Architecture: Paper-faithful DCGAN
  - Generator: z -> 2x2 -> 4x4 -> 8x8 -> 16x16 -> 32x32 (5 conv-transpose layers)
  - Discriminator: 32x32 -> ... -> 1x1 logit (fully convolutional, no Linear)
  - Kernels: 4x4 throughout (NOT 5x5 like the GitHub ref)
  - BN order: Conv -> BN -> Activation (correct order)
  - Weight init: N(0, 0.02) for Conv, N(1.0, 0.02) for BN gamma (per paper)
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Architecture                                                          #
    # ------------------------------------------------------------------ #
    z_dim: int = 100          # Latent vector dimension (paper: 100)
    image_size: int = 32      # CIFAR-10 is 32x32
    image_channels: int = 3   # RGB

    # ------------------------------------------------------------------ #
    # Training                                                              #
    # ------------------------------------------------------------------ #
    batch_size: int = 128         # Paper: 128
    total_steps: int = 50_000     # Reduced from 100k to cut training time in half
    lr_G: float = 2e-4            # Paper: 0.0002
    lr_D: float = 2e-4            # Paper: 0.0002

    # Adam betas — PAPER CRITICAL: beta1=0.5, NOT 0.9
    # The paper found beta1=0.9 caused training oscillation.
    # beta2=0.999 is PyTorch default and standard modern practice.
    betas: Tuple[float, float] = (0.5, 0.999)

    n_dis: int = 1        # D update steps per G update step (1 for standard DCGAN)
    loss: str = "bce"     # Loss type: "bce" (BCE with logits, paper-faithful)

    # LR schedule: linear decay from initial lr to 0 over total_steps.
    # Not in the original paper, but used in the reference GitHub and helps
    # stabilize late-stage training. Set to False to use constant LR.
    use_lr_decay: bool = True

    # ------------------------------------------------------------------ #
    # Evaluation                                                            #
    # ------------------------------------------------------------------ #
    eval_step: int = 5_000    # Compute FID + IS every N steps
    num_images: int = 50_000  # Images to generate for FID/IS (standard: 50k)

    # ------------------------------------------------------------------ #
    # Logging / Sampling                                                    #
    # ------------------------------------------------------------------ #
    sample_step: int = 500    # Save a sample image grid every N steps
    sample_size: int = 64     # Number of images in each sample grid

    # ------------------------------------------------------------------ #
    # Paths                                                                 #
    # ------------------------------------------------------------------ #
    logdir: str = "./logs/DCGAN_CIFAR10"
    # Precomputed Inception activations for CIFAR-10 train set.
    # If this file doesn't exist, train.py and evaluate.py will automatically
    # compute it from CIFAR-10 on first run (~2 min on GPU, ~30 min on CPU)
    # and save it here. No manual download needed.
    fid_cache: str = "./stats/cifar10.train.npz"
    data_dir: str = "./data"   # CIFAR-10 will be downloaded here automatically

    # ------------------------------------------------------------------ #
    # Reproducibility                                                       #
    # ------------------------------------------------------------------ #
    seed: int = 0

    # ------------------------------------------------------------------ #
    # System                                                                #
    # ------------------------------------------------------------------ #
    # Number of DataLoader workers.
    # Kaggle Linux GPU: set to 4.
    # Windows local: set to 0 (Windows multiprocessing can cause issues).
    num_workers: int = 4
