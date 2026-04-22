import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_fidelity
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from config import Config
from dcgan import Generator32, Discriminator32

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_cifar10_dataloader(cfg):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    dataset = datasets.CIFAR10(root=cfg.data_dir,train=True,download=True,transform=transform,)
    
    return DataLoader(dataset,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,drop_last=True,pin_memory=True,)

def infinite_dataloader(dataloader):
    while True:
        for batch, _ in dataloader:
            yield batch

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_real,pred_fake = None,):
        if pred_fake is not None:
            loss_real = self._loss(pred_real, torch.ones_like(pred_real))
            loss_fake = self._loss(pred_fake, torch.zeros_like(pred_fake))
            return loss_real + loss_fake
        else:
            return self._loss(pred_real, torch.ones_like(pred_real))

@torch.no_grad()
def generate_images_for_eval(G,device,z_dim,num_images,batch_size,):
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



def compute_fid_is(G, device, cfg):
    temp_dir = "./temp_train_eval_images"
    G.eval()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print("  Generating images to temp folder for torch-fidelity...")
    generated = 0
    while generated < cfg.num_images:
        n = min(cfg.batch_size, cfg.num_images - generated)
        with torch.no_grad():
            z = torch.randn(n, cfg.z_dim, device=device)
            imgs = G(z).cpu()
            imgs = (imgs + 1.0) / 2.0
            
        for i in range(n):
            img_path = os.path.join(temp_dir, f"img_{generated + i:06d}.png")
            save_image(imgs[i], img_path)
            
        generated += n
    G.train()

    print("  Computing metrics...")
    metrics = torch_fidelity.calculate_metrics(
        input1=temp_dir,
        input2="cifar10-train",
        cuda=True if device.type == "cuda" else False,
        isc=True,
        fid=True,
        verbose=False,
    )
    
    shutil.rmtree(temp_dir)
    
    return metrics['inception_score_mean'], metrics['inception_score_std'], metrics['frechet_inception_distance']


def train(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(cfg.logdir, "sample"), exist_ok=True)
    os.makedirs(os.path.join(cfg.logdir, "checkpoints"), exist_ok=True)

    writer = SummaryWriter(log_dir=cfg.logdir)

    dataloader = get_cifar10_dataloader(cfg)
    data_iter = infinite_dataloader(dataloader)
    steps_per_epoch = len(dataloader)

    G = Generator32(z_dim=cfg.z_dim).to(device)
    D = Discriminator32().to(device)

    loss_fn = BCELoss()

    optim_G = optim.Adam(G.parameters(), lr=cfg.lr_G, betas=cfg.betas)
    optim_D = optim.Adam(D.parameters(), lr=cfg.lr_D, betas=cfg.betas)

    if cfg.use_lr_decay:
        sched_G = optim.lr_scheduler.LambdaLR(
            optim_G, lr_lambda=lambda step: 1.0 - step / cfg.total_steps
        )
        sched_D = optim.lr_scheduler.LambdaLR(
            optim_D, lr_lambda=lambda step: 1.0 - step / cfg.total_steps
        )
    else:
        sched_G = sched_D = None


    real_batch = next(data_iter).to(device)
    real_grid = (make_grid(real_batch[:cfg.sample_size]) + 1.0) / 2.0
    writer.add_image("real_samples", real_grid, global_step=0)
    save_image(real_grid, os.path.join(cfg.logdir, "real_samples.png"))
    print("Saved real sample grid.")

    for step in range(1, cfg.total_steps + 1):

        for _ in range(cfg.n_dis):
            
            real = next(data_iter).to(device)

            with torch.no_grad():
                z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
                fake = G(z)

            pred_real = D(real)
            pred_fake = D(fake)

            loss_D = loss_fn(pred_real, pred_fake)

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
        pred_fake_for_G = D(G(z))

        loss_G = loss_fn(pred_fake_for_G)

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if cfg.use_lr_decay:
            sched_G.step()
            sched_D.step()

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

#evaluation
        if step % cfg.eval_step == 0:
            print(f"\n--- Evaluation at step {step} ---")
            print(f"Generating {cfg.num_images:,} images for FID/IS...")

            IS_mean, IS_std, FID = compute_fid_is(G, device, cfg)

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

    writer.close()
    print("\nTraining complete.")
    print(f"Logs saved to: {cfg.logdir}")
    print(f"Sample grids:  {cfg.logdir}/sample/")
    print(f"Checkpoints:   {cfg.logdir}/checkpoints/")

if __name__ == "__main__":
    cfg = Config()
    train(cfg)
