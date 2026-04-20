"""
train.py
--------
DCGAN training loop for CIFAR-10.

Usage (on Kaggle or local):
    python train.py

What this does:
  1. Loads CIFAR-10 with proper augmentation and normalization to [-1, 1].
  2. Trains Generator and Discriminator with BCE-with-logits loss.
  3. Every `eval_step` steps: generates 50k images, computes FID + IS, saves checkpoint.
  4. Every `sample_step` steps: saves a sample image grid to logdir/sample/.
  5. Logs loss, FID, and IS to TensorBoard.

Training loop design (step-based, not epoch-based):
  - An infinite dataloader wraps CIFAR-10 so we never run out of data.
  - This is the standard approach for GAN training to decouple from dataset size.

Key hyperparameters (all in config.py):
  - lr = 0.0002,  betas = (0.5, 0.999)  — Adam with paper-spec beta1
  - batch_size = 128,  z_dim = 100
  - linear LR decay to 0 over total_steps
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from config import Config
from dcgan import Generator32, Discriminator32


# --------------------------------------------------------------------------- #
# Reproducibility                                                               #
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: Full determinism requires these two lines, but they slow down training.
    # Uncomment if exact bit-for-bit reproducibility is required.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Data                                                                          #
# --------------------------------------------------------------------------- #

def get_cifar10_dataloader(cfg: Config) -> DataLoader:
    """
    Build the CIFAR-10 DataLoader.

    Transforms:
      - RandomHorizontalFlip: standard augmentation for CIFAR-10
      - ToTensor: converts PIL [0,255] to float [0.0, 1.0]
      - Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)): maps [0,1] -> [-1,1]
        This matches the Tanh output range of the Generator.

    drop_last=True: ensures every batch has exactly batch_size samples.
    This prevents shape surprises at the end of an epoch.
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    dataset = datasets.CIFAR10(
        root=cfg.data_dir,
        train=True,       # Use the training split (50,000 images)
        download=True,    # Downloads automatically if not present
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,  # Speeds up CPU -> GPU transfer on Kaggle
    )


def infinite_dataloader(dataloader: DataLoader):
    """
    Wraps a DataLoader to cycle indefinitely.
    Needed for step-based training: we step through the data
    as many times as needed without worrying about epochs.
    """
    while True:
        for batch, _ in dataloader:
            yield batch


# --------------------------------------------------------------------------- #
# Loss                                                                          #
# --------------------------------------------------------------------------- #

class BCELoss(nn.Module):
    """
    BCE-with-Logits loss for GAN training.

    The Discriminator outputs raw logits (no sigmoid).
    nn.BCEWithLogitsLoss applies sigmoid internally for numerical stability.

    Forward call conventions:
      - Discriminator update: pass both pred_real and pred_fake
          -> loss = BCE(D(real), 1) + BCE(D(fake), 0)
      - Generator update:     pass only pred_fake
          -> loss = BCE(D(G(z)), 1)   [G wants D to think fakes are real]
    """

    def __init__(self) -> None:
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor = None,
    ) -> torch.Tensor:
        if pred_fake is not None:
            # Discriminator loss: real -> 1, fake -> 0
            loss_real = self._loss(pred_real, torch.ones_like(pred_real))
            loss_fake = self._loss(pred_fake, torch.zeros_like(pred_fake))
            return loss_real + loss_fake
        else:
            # Generator loss: G wants D to predict fake as real -> target = 1
            return self._loss(pred_real, torch.ones_like(pred_real))


# --------------------------------------------------------------------------- #
# Evaluation helpers                                                            #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate_images_for_eval(
    G: nn.Module,
    device: torch.device,
    z_dim: int,
    num_images: int,
    batch_size: int,
) -> torch.Tensor:
    """
    Generate `num_images` images from the Generator for FID/IS computation.

    Returns:
        Tensor of shape (num_images, 3, 32, 32) with pixel values in [0, 1].
        (FID/IS libraries expect [0, 1], not [-1, 1].)
    """
    G.eval()
    images = []

    for start in range(0, num_images, batch_size):
        end = min(start + batch_size, num_images)
        z = torch.randn(end - start, z_dim, device=device)
        imgs = G(z).cpu()          # (batch, 3, 32, 32) in [-1, 1]
        imgs = (imgs + 1.0) / 2.0  # Rescale to [0, 1]
        images.append(imgs)

    G.train()
    return torch.cat(images, dim=0)  # (num_images, 3, 32, 32)


def ensure_fid_cache(cfg: Config, device: torch.device) -> tuple:
    """
    Ensure FID reference statistics are available (either from cache or computed fresh).

    If cfg.fid_cache exists as a .npz file, load mu and sigma from it.
    Otherwise, compute them from CIFAR-10 and save to disk.

    This is a one-time cost (~2-3 min on GPU, ~30 min on CPU).
    After that, every FID computation reuses the cached stats instantly.

    Returns:
        (mu, sigma) tuple of numpy arrays for the real image distribution.
    """
    from metrics import compute_real_stats

    if os.path.exists(cfg.fid_cache):
        print(f"FID cache found: {cfg.fid_cache}")
        data = np.load(cfg.fid_cache)
        return data["mu"], data["sigma"]

    print(f"FID cache not found at: {cfg.fid_cache}")
    print("Computing CIFAR-10 Inception statistics (one-time cost)...")
    print("This takes ~2-3 min on GPU, ~30 min on CPU.\n")

    # Load CIFAR-10 WITHOUT augmentation — we need the raw images
    dataset = datasets.CIFAR10(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),  # Raw [0, 1], no augmentation
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    # Collect all images
    all_images = []
    for imgs, _ in loader:
        all_images.append(imgs)
    all_images = torch.cat(all_images, dim=0)  # (50000, 3, 32, 32) in [0, 1]
    print(f"Loaded {all_images.shape[0]} CIFAR-10 images.")

    # Compute Inception features and stats
    mu, sigma = compute_real_stats(all_images, device, batch_size=64)

    # Save to disk for reuse
    os.makedirs(os.path.dirname(cfg.fid_cache), exist_ok=True)
    np.savez(cfg.fid_cache, mu=mu, sigma=sigma)
    print(f"\nFID cache saved to: {cfg.fid_cache}")
    print("This file will be reused for all future FID computations.\n")

    return mu, sigma


def compute_fid_is(G, device, cfg, real_mu, real_sigma):
    """
    Compute Inception Score and FID using our custom metrics.py.

    No external dependencies — uses torchvision InceptionV3 + scipy.

    Args:
        G: Generator model.
        device: torch device.
        cfg: Config dataclass.
        real_mu: (2048,) precomputed mean of real image features.
        real_sigma: (2048, 2048) precomputed covariance of real image features.

    Returns:
        IS_mean (float), IS_std (float), FID (float)
    """
    from metrics import compute_fid_and_is

    # Generate images for evaluation
    imgs = generate_images_for_eval(
        G, device, cfg.z_dim, cfg.num_images, cfg.batch_size
    )

    # Compute both metrics
    IS_mean, IS_std, FID = compute_fid_and_is(
        imgs, real_mu, real_sigma, device, batch_size=64
    )

    return IS_mean, IS_std, FID


# --------------------------------------------------------------------------- #
# Training loop                                                                 #
# --------------------------------------------------------------------------- #

def train(cfg: Config) -> None:
    # ---- Setup ------------------------------------------------------------ #
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Total steps: {cfg.total_steps:,}  |  Batch size: {cfg.batch_size}")
    print(f"z_dim: {cfg.z_dim}  |  lr_G: {cfg.lr_G}  |  lr_D: {cfg.lr_D}")
    print(f"Adam betas: {cfg.betas}  |  n_dis: {cfg.n_dis}\n")

    os.makedirs(os.path.join(cfg.logdir, "sample"), exist_ok=True)
    os.makedirs(os.path.join(cfg.logdir, "checkpoints"), exist_ok=True)

    writer = SummaryWriter(log_dir=cfg.logdir)

    # ---- FID cache (one-time computation) --------------------------------- #
    # If ./stats/cifar10.train.npz doesn't exist, this computes it from
    # CIFAR-10 automatically. No manual download needed.
    real_mu, real_sigma = ensure_fid_cache(cfg, device)

    # ---- Data ------------------------------------------------------------- #
    dataloader = get_cifar10_dataloader(cfg)
    data_iter = infinite_dataloader(dataloader)
    steps_per_epoch = len(dataloader)

    # ---- Models ----------------------------------------------------------- #
    G = Generator32(z_dim=cfg.z_dim).to(device)
    D = Discriminator32().to(device)

    # ---- Loss ------------------------------------------------------------- #
    loss_fn = BCELoss()

    # ---- Optimizers ------------------------------------------------------- #
    # Adam with beta1=0.5 (DCGAN paper), beta2=0.999 (standard)
    optim_G = optim.Adam(G.parameters(), lr=cfg.lr_G, betas=cfg.betas)
    optim_D = optim.Adam(D.parameters(), lr=cfg.lr_D, betas=cfg.betas)

    # ---- LR Schedulers ---------------------------------------------------- #
    # Linear decay: lr goes from initial value to 0 over total_steps.
    # lambda(step) = 1 - step / total_steps
    if cfg.use_lr_decay:
        sched_G = optim.lr_scheduler.LambdaLR(
            optim_G, lr_lambda=lambda step: 1.0 - step / cfg.total_steps
        )
        sched_D = optim.lr_scheduler.LambdaLR(
            optim_D, lr_lambda=lambda step: 1.0 - step / cfg.total_steps
        )
    else:
        sched_G = sched_D = None

    # ---- Fixed noise for consistent sample grids -------------------------- #
    # We use the same z across all sample steps so we can visually track
    # how the same latent points evolve during training.
    fixed_z = torch.randn(cfg.sample_size, cfg.z_dim, device=device)

    # Log a real image grid so we can compare generated vs real
    real_batch = next(data_iter).to(device)
    real_grid = (make_grid(real_batch[:cfg.sample_size]) + 1.0) / 2.0
    writer.add_image("real_samples", real_grid, global_step=0)
    save_image(real_grid, os.path.join(cfg.logdir, "real_samples.png"))
    print("Saved real sample grid.")

    # ---- Training loop ---------------------------------------------------- #
    print("\nStarting training...\n")

    for step in range(1, cfg.total_steps + 1):

        # ------------------------------------------------------------------- #
        # Step 1: Train Discriminator                                           #
        # n_dis=1 for standard DCGAN. Could be >1 for WGAN-style training.     #
        # ------------------------------------------------------------------- #
        for _ in range(cfg.n_dis):
            # Sample real images from the infinite data iterator
            real = next(data_iter).to(device)

            # Generate fake images — detach so G gradients don't flow through D step
            with torch.no_grad():
                z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
                fake = G(z)

            # Forward pass through D
            pred_real = D(real)
            pred_fake = D(fake)

            # D loss: maximize log D(x) + log(1 - D(G(z)))
            # With BCEWithLogits: minimize -[log D(x) + log(1 - D(G(z)))]
            loss_D = loss_fn(pred_real, pred_fake)

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # ------------------------------------------------------------------- #
        # Step 2: Train Generator                                           #
        # ------------------------------------------------------------------- #
        # Use standard batch_size for G update.
        # DO NOT use D.eval() here. It causes BatchNorm running stats to shift
        # the activations so much that gradients vanish (loss_G = 0.0000).
        z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
        pred_fake_for_G = D(G(z))

        # G loss: minimize -log D(G(z))  [wants D to predict fake as real]
        loss_G = loss_fn(pred_fake_for_G)

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        # ------------------------------------------------------------------- #
        # Step 3: LR decay step                                                 #
        # ------------------------------------------------------------------- #
        if cfg.use_lr_decay:
            sched_G.step()
            sched_D.step()

        # ------------------------------------------------------------------- #
        # Logging: losses to TensorBoard                                        #
        # ------------------------------------------------------------------- #
        writer.add_scalar("Loss/Discriminator", loss_D.item(), step)
        writer.add_scalar("Loss/Generator",     loss_G.item(), step)

        if step % 100 == 0:
            current_lr = optim_G.param_groups[0]["lr"]
            current_epoch = step // steps_per_epoch
            total_epochs = cfg.total_steps // steps_per_epoch
            
            print(
                f"Epoch [{current_epoch:>3}/{total_epochs}] "
                f"Step [{step:>6}/{cfg.total_steps}] "
                f"loss_D: {loss_D.item():.4f}  "
                f"loss_G: {loss_G.item():.4f}  "
                f"lr: {current_lr:.6f}"
            )

        # ------------------------------------------------------------------- #
        # Sample grid: visualize generator progress                             #
        # ------------------------------------------------------------------- #
        if step == 1 or step % cfg.sample_step == 0:
            G.eval()
            with torch.no_grad():
                fake_samples = G(fixed_z).cpu()
            G.train()

            grid = (make_grid(fake_samples, nrow=8) + 1.0) / 2.0
            writer.add_image("generated_samples", grid, global_step=step)
            sample_path = os.path.join(cfg.logdir, "sample", f"step_{step:07d}.png")
            save_image(grid, sample_path)

        # ------------------------------------------------------------------- #
        # Evaluation: FID + IS (expensive — runs every eval_step steps)        #
        # ------------------------------------------------------------------- #
        if step % cfg.eval_step == 0:
            print(f"\n--- Evaluation at step {step} ---")
            print(f"Generating {cfg.num_images:,} images for FID/IS...")

            IS_mean, IS_std, FID = compute_fid_is(G, device, cfg, real_mu, real_sigma)

            if IS_mean is not None:
                print(
                    f"Step {step:>6}: "
                    f"IS = {IS_mean:.3f} ± {IS_std:.5f}  |  FID = {FID:.3f}"
                )
                writer.add_scalar("Metrics/Inception_Score",     IS_mean, step)
                writer.add_scalar("Metrics/Inception_Score_std", IS_std,  step)
                writer.add_scalar("Metrics/FID",                 FID,     step)

            # Save checkpoint
            ckpt_path = os.path.join(
                cfg.logdir, "checkpoints", f"ckpt_step_{step:07d}.pt"
            )
            torch.save(
                {
                    "step":      step,
                    "net_G":     G.state_dict(),
                    "net_D":     D.state_dict(),
                    "optim_G":   optim_G.state_dict(),
                    "optim_D":   optim_D.state_dict(),
                    "sched_G":   sched_G.state_dict() if sched_G else None,
                    "sched_D":   sched_D.state_dict() if sched_D else None,
                    "IS_mean":   IS_mean,
                    "IS_std":    IS_std,
                    "FID":       FID,
                    "config":    cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"Checkpoint saved: {ckpt_path}\n")

            # Also overwrite a "latest" checkpoint for easy resuming
            torch.save(
                {
                    "step":    step,
                    "net_G":   G.state_dict(),
                    "net_D":   D.state_dict(),
                    "optim_G": optim_G.state_dict(),
                    "optim_D": optim_D.state_dict(),
                    "sched_G": sched_G.state_dict() if sched_G else None,
                    "sched_D": sched_D.state_dict() if sched_D else None,
                    "config":  cfg.__dict__,
                },
                os.path.join(cfg.logdir, "model_latest.pt"),
            )

    # ---- Training complete ------------------------------------------------ #
    writer.close()
    print("\nTraining complete.")
    print(f"Logs saved to: {cfg.logdir}")
    print(f"Sample grids:  {cfg.logdir}/sample/")
    print(f"Checkpoints:   {cfg.logdir}/checkpoints/")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    cfg = Config()
    train(cfg)
