import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class GeneDataset(Dataset):
    def __init__(self, x, c):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.c[idx]

class Encoder(nn.Module):
    def __init__(self, expr_dim, cond_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(expr_dim + cond_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logv = nn.Linear(128, latent_dim)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x, c):
        h = torch.cat([x, c], dim=1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        h = self.act(self.fc4(h))
        return self.fc_mu(h), self.fc_logv(h)

class Decoder(nn.Module):
    def __init__(self, expr_dim, cond_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + cond_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, expr_dim)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, z, c):
        h = self.act(self.fc1(torch.cat([z, c], dim=1)))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        return torch.sigmoid(self.fc4(h))

def reparam(mu, logv):
    std = (0.5 * logv).exp()
    eps = torch.randn_like(std)
    return mu + eps * std

def weighted_mse(x_hat, x_true, weights):
    return ((weights * (x_hat - x_true) ** 2).mean())