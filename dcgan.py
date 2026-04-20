"""
dcgan.py
--------
DCGAN architecture for CIFAR-10 (32x32 images).

Design choices vs the reference GitHub (w86763777/pytorch-gan-collections):
  - Kernel size 4x4 in Discriminator (GitHub uses 5x5)
  - Correct BN order: Conv -> BN -> Activation (GitHub has BN after activation)
  - Fully convolutional Discriminator — final Conv2d collapses to 1x1 logit
    (GitHub uses nn.Linear, which the DCGAN paper says to avoid)
  - Explicit weight initialization from N(0, 0.02) (GitHub has NO weight init)

Generator spatial flow (z_dim=100, image_size=32):
  z  : (B, 100)
  -> reshape -> (B, 100, 1, 1)
  -> ConvT(100, 1024, k=2, s=1, p=0) -> (B, 1024, 2, 2)   + BN + ReLU
  -> ConvT(1024, 512, k=4, s=2, p=1) -> (B,  512, 4, 4)   + BN + ReLU
  -> ConvT(512,  256, k=4, s=2, p=1) -> (B,  256, 8, 8)   + BN + ReLU
  -> ConvT(256,  128, k=4, s=2, p=1) -> (B,  128,16,16)   + BN + ReLU
  -> ConvT(128,    3, k=4, s=2, p=1) -> (B,    3,32,32)   + Tanh

Discriminator spatial flow (image_size=32):
  x  : (B, 3, 32, 32)
  -> Conv(3,   64, k=4, s=2, p=1) -> (B,  64, 16, 16)  + LeakyReLU(0.2)
  -> Conv(64, 128, k=4, s=2, p=1) -> (B, 128,  8,  8)  + BN + LeakyReLU(0.2)
  -> Conv(128,256, k=4, s=2, p=1) -> (B, 256,  4,  4)  + BN + LeakyReLU(0.2)
  -> Conv(256,512, k=4, s=2, p=1) -> (B, 512,  2,  2)  + BN + LeakyReLU(0.2)
  -> Conv(512,  1, k=2, s=1, p=0) -> (B,   1,  1,  1)  (raw logit)
  -> flatten -> (B, 1)
"""

import torch
import torch.nn as nn


__all__ = ["Generator32", "Discriminator32"]


# --------------------------------------------------------------------------- #
# Weight initialization                                                         #
# --------------------------------------------------------------------------- #

def weights_init(m: nn.Module) -> None:
    """
    Initialize weights as specified in the DCGAN paper:
      - Conv / ConvTranspose weights: N(0, 0.02)
      - BatchNorm gamma (weight):     N(1.0, 0.02)   [near-identity start]
      - BatchNorm beta  (bias):       0
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # Covers both Conv2d and ConvTranspose2d
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0)


# --------------------------------------------------------------------------- #
# Generator                                                                     #
# --------------------------------------------------------------------------- #

class Generator(nn.Module):
    """
    Fully convolutional generator.

    Args:
        z_dim (int): Dimensionality of the latent noise vector.
        M     (int): Controls the initial spatial size after the first
                     ConvTranspose2d. For 32x32 output: M=2 (gives 2x2 -> 32x32).
    """

    def __init__(self, z_dim: int, M: int) -> None:
        super().__init__()
        self.z_dim = z_dim

        self.main = nn.Sequential(
            # (B, z_dim, 1, 1) -> (B, 512, M, M)
            # For CIFAR-10 (32x32), M=4: projects to a 4x4 spatial map.
            nn.ConvTranspose2d(z_dim, 512, kernel_size=M, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # (B, 512, M, M) -> (B, 256, 2M, 2M)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (B, 256, 2M, 2M) -> (B, 128, 4M, 4M)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (B, 128, 4M, 4M) -> (B, 3, 8M, 8M)
            # For M=4: output is (B, 3, 32, 32) ✓
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),   # Output in [-1, 1] to match normalized input images
        )

        # Apply paper weight initialization to all submodules
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent noise tensor of shape (B, z_dim).
        Returns:
            Generated images of shape (B, 3, 32, 32) in range [-1, 1].
        """
        # Reshape flat noise to (B, z_dim, 1, 1) spatial seed
        return self.main(z.view(-1, self.z_dim, 1, 1))


# --------------------------------------------------------------------------- #
# Discriminator                                                                 #
# --------------------------------------------------------------------------- #

class Discriminator(nn.Module):
    """
    Fully convolutional discriminator. Outputs a raw logit (no sigmoid).
    Use BCEWithLogitsLoss, not BCELoss, to pair with this discriminator.

    Args:
        M (int): Input image size. For CIFAR-10: M=32.
                 Used to compute the final Conv2d kernel size (M // 16)
                 that collapses spatial dims to 1x1.
    """

    def __init__(self, M: int) -> None:
        super().__init__()

        self.main = nn.Sequential(
            # No BN on the first layer — paper guideline.
            # Using bias=True here because there is no BatchNorm to follow.
            # (B, 3, M, M) -> (B, 64, M/2, M/2)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 64, M/2, M/2) -> (B, 128, M/4, M/4)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (B, 128, M/4, M/4) -> (B, 256, M/8, M/8)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Final conv: collapses (M/8 x M/8) spatial map to (1 x 1).
            # For M=32: kernel_size = 32//8 = 4 -> (B, 256, 4, 4) -> (B, 1, 1, 1)
            # No BN, no activation — raw logit output.
            nn.Conv2d(256, 1, kernel_size=M // 8, stride=1, padding=0, bias=True),
        )

        # Apply paper weight initialization
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (B, 3, 32, 32) in range [-1, 1].
        Returns:
            Raw logits of shape (B, 1). Do NOT apply sigmoid here.
        """
        x = self.main(x)
        # Flatten the (B, 1, 1, 1) output to (B, 1)
        return torch.flatten(x, start_dim=1)


# --------------------------------------------------------------------------- #
# Concrete 32x32 models (CIFAR-10)                                             #
# --------------------------------------------------------------------------- #

class Generator32(Generator):
    """Generator for 32x32 images (CIFAR-10)."""
    def __init__(self, z_dim: int = 100) -> None:
        # M=4: first ConvTranspose2d projects to a 4x4 spatial map,
        # then 3 upsampling stages bring it to 32x32.
        super().__init__(z_dim=z_dim, M=4)


class Discriminator32(Discriminator):
    """Discriminator for 32x32 images (CIFAR-10)."""
    def __init__(self) -> None:
        # M=32: final Conv2d uses kernel_size = 32//8 = 4
        super().__init__(M=32)


# --------------------------------------------------------------------------- #
# Quick shape sanity check (run this file directly to verify)                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running shape check on: {device}\n")

    z_dim = 100
    batch_size = 4

    G = Generator32(z_dim=z_dim).to(device)
    D = Discriminator32().to(device)

    # Test Generator
    z = torch.randn(batch_size, z_dim, device=device)
    fake = G(z)
    assert fake.shape == (batch_size, 3, 32, 32), f"G output shape mismatch: {fake.shape}"
    print(f"Generator output:      {fake.shape}  [OK]   (expected: [{batch_size}, 3, 32, 32])")

    # Test Discriminator
    logits = D(fake)
    assert logits.shape == (batch_size, 1), f"D output shape mismatch: {logits.shape}"
    print(f"Discriminator output:  {logits.shape}     [OK]   (expected: [{batch_size}, 1])")

    # Parameter counts
    G_params = sum(p.numel() for p in G.parameters())
    D_params = sum(p.numel() for p in D.parameters())
    print(f"\nGenerator parameters:     {G_params:,}")
    print(f"Discriminator parameters: {D_params:,}")
    print("\nAll shape checks passed [OK]")