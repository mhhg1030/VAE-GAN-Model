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
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logv = nn.Linear(256, latent_dim)

        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, c):
        h = torch.cat([x, c], dim=1)
        h = self.dropout(self.act(self.bn1(self.fc1(h))))
        h = self.dropout(self.act(self.bn2(self.fc2(h))))
        h = self.act(self.bn3(self.fc3(h)))
        return self.fc_mu(h), self.fc_logv(h)
    
class Decoder(nn.Module):
    def __init__(self, expr_dim, cond_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + cond_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512 + cond_dim, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512 + cond_dim, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc_out = nn.Linear(256, expr_dim)

        self.act = nn.LeakyReLU(0.2)

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)

        h1 = self.act(self.bn1(self.fc1(x)))
        h1 = torch.cat([h1, c], dim=1)

        h2 = self.act(self.bn2(self.fc2(h1)))
        h2 = torch.cat([h2, c], dim=1)

        h3 = self.act(self.bn3(self.fc3(h2)))
        return self.fc_out(h3)

def reparam(mu, logv):
    std = (0.5 * logv).exp()
    eps = torch.randn_like(std)
    return mu + eps * std

def combined_loss(pred, target, weights):
    mse = ((pred - target)**2).mean(dim=0)
    mae = (pred - target).abs().mean(dim=0)
    return (0.5 * mse + 0.5 * mae) @ weights


def kl_anneal(epoch, warmup_epochs=300):
    return min(1.0, epoch / warmup_epochs)
