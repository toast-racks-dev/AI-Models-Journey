import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, z_dim, M):
        super().__init__()
        self.z_dim = z_dim
        self.main = nn.Sequential(
            # (B, z_dim, 1, 1) -> (B, 512, 4, 4)
            nn.ConvTranspose2d(z_dim, 512, kernel_size=M, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # (B, 512, 4, 4) -> (B, 256, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (B, 256, 8, 8) -> (B, 128, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (B, 128, 16, 16) -> (B, 3, 32, 32)
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, z):
        # Reshape flat noise to (B, z_dim, 1, 1) 
        return self.main(z.view(-1, self.z_dim, 1, 1))

class Discriminator(nn.Module):
    def __init__(self, M):
        super().__init__()

        self.main = nn.Sequential(
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

            # Input: (B, 256, M/8, M/8) -> (B, 256, 4, 4) for CIFAR
            # Final conv: Maintains a 4x4 spatial map instead of collapsing to 1x1.
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.apply(weights_init)

    def forward(self, x):
        x = self.main(x)
        return torch.flatten(x, start_dim=1)

class Generator32(Generator):
    def __init__(self, z_dim=100):
        super().__init__(z_dim=z_dim, M=4)
        
class Discriminator32(Discriminator):
    def __init__(self):
        super().__init__(M=32)

