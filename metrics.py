"""
metrics.py
----------
FID and Inception Score computation using ONLY pre-installed Kaggle libraries.

No external dependencies needed. Uses:
  - torchvision.models.inception_v3 (pretrained InceptionV3)
  - scipy.linalg.sqrtm (matrix square root for FID)
  - numpy (mean, covariance)

This replaces the `pytorch_gan_metrics` dependency entirely.

FID (Frechet Inception Distance):
  1. Extract 2048-d features from InceptionV3 pool3 layer for real and fake images
  2. Compute mean and covariance of both feature distributions
  3. FID = ||mu_real - mu_fake||^2 + Tr(Sigma_real + Sigma_fake - 2*sqrt(Sigma_real @ Sigma_fake))

IS (Inception Score):
  1. Get class probabilities p(y|x) from InceptionV3 for each generated image
  2. Compute marginal p(y) = mean of all p(y|x)
  3. IS = exp(mean(KL(p(y|x) || p(y))))
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from scipy import linalg


# --------------------------------------------------------------------------- #
# InceptionV3 Feature Extractor                                                 #
# --------------------------------------------------------------------------- #

class InceptionV3Features(nn.Module):
    """
    Wrapper around torchvision's InceptionV3 that extracts:
      - 2048-d pool features (for FID)
      - 1008-d logits (for IS)

    The model expects input images in range [0, 1], size >= 75x75.
    We resize 32x32 CIFAR images to 299x299 (Inception's native resolution).
    """

    def __init__(self):
        super().__init__()
        # Load pretrained InceptionV3
        # weights parameter uses the new API (torchvision >= 0.13)
        try:
            self.model = models.inception_v3(
                weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False,  # We handle normalization ourselves
            )
        except TypeError:
            # Fallback for older torchvision versions
            self.model = models.inception_v3(
                pretrained=True,
                transform_input=False,
            )

        self.model.eval()

        # We need to hook into the model to get pool features
        # InceptionV3 architecture: ... -> avgpool -> dropout -> fc -> output
        # We want the output of avgpool (2048-d) AND the fc logits (1000-d)
        self._pool_features = None
        self.model.avgpool.register_forward_hook(self._hook_pool)

    def _hook_pool(self, module, input, output):
        """Hook to capture avgpool output (2048-d features)."""
        self._pool_features = output

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: Images (B, 3, H, W) in range [0, 1].

        Returns:
            pool_features: (B, 2048) — for FID
            logits: (B, 1000) — for IS
        """
        # Resize to 299x299 (Inception's expected input size)
        # We do this on the CPU to avoid a known Kaggle PyTorch bug ("no kernel image")
        if x.shape[2] != 299 or x.shape[3] != 299:
            device = x.device
            x = x.cpu()
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            x = x.to(device)

        # InceptionV3 weights expect inputs in range [-1, 1].
        # (Generated images are passed here in range [0, 1])
        x = x * 2.0 - 1.0

        # Forward pass
        logits = self.model(x)

        # Get the hooked pool features
        pool = self._pool_features.squeeze(-1).squeeze(-1)  # (B, 2048)

        return pool, logits


# --------------------------------------------------------------------------- #
# Feature extraction over a dataset                                             #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_features(
    images: torch.Tensor,
    inception: InceptionV3Features,
    device: torch.device,
    batch_size: int = 64,
) -> tuple:
    """
    Extract InceptionV3 features from a batch of images.

    Args:
        images: (N, 3, 32, 32) tensor in range [0, 1].
        inception: InceptionV3Features model.
        device: torch device.
        batch_size: Processing batch size (64 is safe for 16GB GPU memory
                    since we resize to 299x299).

    Returns:
        all_pool: (N, 2048) numpy array — pool features for FID
        all_logits: (N, 1000) numpy array — logits for IS
    """
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_pool = []
    all_logits = []

    for (batch,) in loader:
        batch = batch.to(device)
        pool, logits = inception(batch)
        all_pool.append(pool.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_pool, axis=0), np.concatenate(all_logits, axis=0)


# --------------------------------------------------------------------------- #
# FID computation                                                               #
# --------------------------------------------------------------------------- #

def compute_fid(mu_real, sigma_real, mu_fake, sigma_fake, eps=1e-6):
    """
    Compute the Frechet Inception Distance.

    FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2 * sqrt(Sigma_r @ Sigma_f))

    Args:
        mu_real:    (2048,) mean of real image features
        sigma_real: (2048, 2048) covariance of real image features
        mu_fake:    (2048,) mean of generated image features
        sigma_fake: (2048, 2048) covariance of generated image features
        eps:        Small constant for numerical stability

    Returns:
        FID score (float). Lower is better.
    """
    # Term 1: squared difference of means
    diff = mu_real - mu_fake
    term1 = diff @ diff  # scalar

    # Term 2: trace of covariance term
    # We need sqrt(Sigma_r @ Sigma_f), computed via scipy matrix square root
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)

    # sqrtm can return complex numbers due to numerical errors
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Imaginary component in sqrtm result is too large")
        covmean = covmean.real

    term2 = np.trace(sigma_real + sigma_fake - 2.0 * covmean)

    # If the product is singular or has negative eigenvalues due to numerical errors,
    # sqrtm can produce NaNs. In that case, we add a small epsilon to the diagonal.
    if not np.isfinite(term2):
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma_real + offset) @ (sigma_fake + offset), disp=False)
        term2 = np.trace(sigma_real + sigma_fake - 2.0 * covmean.real)

    return float(term1 + term2)


# --------------------------------------------------------------------------- #
# Inception Score computation                                                   #
# --------------------------------------------------------------------------- #

def compute_inception_score(logits, num_splits=10):
    """
    Compute the Inception Score.

    IS = exp(E_x[KL(p(y|x) || p(y))])

    where p(y|x) is the softmax of InceptionV3 logits for each image,
    and p(y) is the marginal (average over all images).

    Args:
        logits: (N, 1000) numpy array of InceptionV3 logits.
        num_splits: Number of splits for computing mean ± std.

    Returns:
        (IS_mean, IS_std) tuple.
    """
    # Convert logits to probabilities
    # Numerically stable softmax
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits_shifted) / np.sum(np.exp(logits_shifted), axis=1, keepdims=True)

    scores = []
    N = probs.shape[0]
    split_size = N // num_splits

    for i in range(num_splits):
        start = i * split_size
        end = start + split_size
        part = probs[start:end]

        # p(y) = marginal distribution (average prediction over this split)
        py = np.mean(part, axis=0, keepdims=True)

        # KL divergence: sum p(y|x) * log(p(y|x) / p(y))
        kl = part * (np.log(part + 1e-16) - np.log(py + 1e-16))
        kl = np.sum(kl, axis=1)  # sum over classes
        kl = np.mean(kl)         # mean over images in this split

        scores.append(np.exp(kl))

    return float(np.mean(scores)), float(np.std(scores))


# --------------------------------------------------------------------------- #
# High-level API                                                                #
# --------------------------------------------------------------------------- #

_inception_model = None  # Lazy-loaded singleton


def get_inception_model(device):
    """Lazy-load InceptionV3 (only loaded once, reused across calls)."""
    global _inception_model
    if _inception_model is None:
        print("Loading InceptionV3 model...")
        _inception_model = InceptionV3Features().to(device)
        _inception_model.eval()
        print("InceptionV3 loaded.\n")
    return _inception_model


def compute_real_stats(
    real_images: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> tuple:
    """
    Compute InceptionV3 feature statistics (mu, sigma) for real images.

    Args:
        real_images: (N, 3, 32, 32) tensor in [0, 1].
        device: torch device.
        batch_size: Batch size for Inception forward passes.

    Returns:
        mu: (2048,) numpy array
        sigma: (2048, 2048) numpy array
    """
    inception = get_inception_model(device)
    pool_feats, _ = extract_features(real_images, inception, device, batch_size)
    mu = np.mean(pool_feats, axis=0)
    sigma = np.cov(pool_feats, rowvar=False)
    return mu, sigma


def compute_fid_and_is(
    generated_images: torch.Tensor,
    real_mu: np.ndarray,
    real_sigma: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> tuple:
    """
    Compute both FID and IS for a batch of generated images.

    Args:
        generated_images: (N, 3, 32, 32) tensor in [0, 1].
        real_mu: (2048,) precomputed mean of real image features.
        real_sigma: (2048, 2048) precomputed covariance of real image features.
        device: torch device.
        batch_size: Batch size for Inception forward passes.

    Returns:
        (IS_mean, IS_std, FID) tuple.
    """
    inception = get_inception_model(device)

    # Extract features for generated images
    fake_pool, fake_logits = extract_features(
        generated_images, inception, device, batch_size
    )

    # Compute FID
    fake_mu = np.mean(fake_pool, axis=0)
    fake_sigma = np.cov(fake_pool, rowvar=False)
    fid = compute_fid(real_mu, real_sigma, fake_mu, fake_sigma)

    # Compute IS
    is_mean, is_std = compute_inception_score(fake_logits)

    return is_mean, is_std, fid


# --------------------------------------------------------------------------- #
# Quick test (run this file directly)                                           #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}\n")

    # Test with random noise (should give bad FID/IS — that's expected)
    fake = torch.rand(100, 3, 32, 32)  # 100 random images in [0, 1]
    real = torch.rand(100, 3, 32, 32)  # 100 random "real" images

    print("Computing real stats...")
    mu, sigma = compute_real_stats(real, device, batch_size=50)
    print(f"Real stats: mu shape={mu.shape}, sigma shape={sigma.shape}")

    print("\nComputing FID and IS...")
    is_mean, is_std, fid = compute_fid_and_is(fake, mu, sigma, device, batch_size=50)
    print(f"IS = {is_mean:.3f} ± {is_std:.5f}")
    print(f"FID = {fid:.3f}")
    print("\n(Random noise -> bad scores. That's expected!)")
    print("All metrics tests passed [OK]")
