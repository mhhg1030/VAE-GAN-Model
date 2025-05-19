#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

def main():
    BASE_DIR = Path(__file__).parent.resolve()
    data_dir = BASE_DIR / "data"
    excel_path = data_dir / "NP-PC Database(Part).xlsx"

    df = pd.read_excel(excel_path, engine="openpyxl")
    df.columns = df.columns.str.strip()

    print("\nEnter condition columns (or 'done'):")
    condition_columns = []
    while True:
        c = input("  Condition column: ").strip()
        if c.lower() == "done": break
        condition_columns.append(c)

    print("\nEnter gene column letters (e.g. L, M, etc) (or 'done'):")
    gene_columns = []
    while True:
        g = input("  Gene column: ").strip()
        if g.lower() == "done": break
        gene_columns.append(g)

    def col_to_idx(name):
        idx = 0
        for ch in name.upper():
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
        return idx - 1

    gene_idx = [col_to_idx(g) for g in gene_columns]
    gene_names = df.columns[gene_idx]
    X_raw = df.iloc[:, gene_idx].values.astype(np.float32)

    # Drop rows where all genes are 0
    mask = ~(X_raw == 0).all(axis=1)
    df = df[mask].reset_index(drop=True)
    X_raw = X_raw[mask]

    df_selected = df[condition_columns].reset_index(drop=True).join(
        pd.DataFrame(X_raw, columns=gene_names))
    df_selected.to_csv(data_dir / "input.csv", index=False)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)
    expr_dim = X.shape[1]
    gene_weights_np = 1.0 / (X_raw.std(axis=0) + 1e-6)
    gene_weights = torch.tensor(gene_weights_np, dtype=torch.float32)

    encoders = {
        col: {v: [1 if i == j else 0 for j in range(len(vals))]
              for i, v in enumerate(vals)}
        for col, vals in ((col, df[col].unique().tolist()) for col in condition_columns)
    }
    C = np.array([
        sum((encoders[col][row[col]] for col in condition_columns), [])
        for _, row in df.iterrows()
    ], dtype=np.float32)
    cond_dim = C.shape[1]

    Xtr, Xv, Ctr, Cv = train_test_split(X, C, test_size=0.2, random_state=42)
    train_loader = DataLoader(GeneDataset(Xtr, Ctr), batch_size=128, shuffle=True)
    val_loader = DataLoader(GeneDataset(Xv, Cv), batch_size=128, shuffle=False)

    latent_dim = 256
    kl_weight = 0.01
    enc = Encoder(expr_dim, cond_dim, latent_dim)
    dec = Decoder(expr_dim, cond_dim, latent_dim)
    opt_e = optim.AdamW(enc.parameters(), lr=5e-4)
    opt_d = optim.AdamW(dec.parameters(), lr=5e-4)

    for epoch in range(1, 501):
        enc.train(); dec.train()
        for x_b, c_b in train_loader:
            mu, logv = enc(x_b, c_b)
            z = reparam(mu, logv)
            x_hat = dec(z, c_b)
            recon_loss = weighted_mse(x_hat, x_b, gene_weights)
            kl_loss = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()
            loss = 10 * recon_loss + kl_weight * kl_loss
            opt_e.zero_grad(); opt_d.zero_grad()
            loss.backward()
            opt_e.step(); opt_d.step()
        print(f"Epoch {epoch:03d} | Recon={recon_loss:.4f} | KL={kl_loss:.4f}")

    enc.eval(); dec.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for x_b, c_b in val_loader:
            mu, logv = enc(x_b, c_b)
            z = reparam(mu, logv)
            x_hat = dec(z, c_b)
            all_t.append(x_b.numpy())
            all_p.append(x_hat.numpy())

    true_inv = scaler.inverse_transform(np.vstack(all_t))
    pred_inv = scaler.inverse_transform(np.vstack(all_p))
    rmse = np.sqrt(mean_squared_error(true_inv, pred_inv))
    r2 = r2_score(true_inv, pred_inv)
    scaled_rmse = np.sqrt(((np.vstack(all_t) - np.vstack(all_p))**2).mean())
    print(f"\nVal RMSE: {rmse:.4f}, R²: {r2:.4f}, Scaled RMSE: {scaled_rmse:.4f}\n")

    # === Save predictions to output.csv ===
    with torch.no_grad():
        generated = []
        for c_vec in torch.tensor(C):
            z = torch.randn(1, latent_dim)
            x_pred = dec(z, c_vec.unsqueeze(0)).squeeze(0).numpy()
            generated.append(x_pred)
    df_pred = pd.DataFrame(scaler.inverse_transform(generated), columns=gene_names)
    df_out = pd.concat([df[condition_columns].reset_index(drop=True), df_pred], axis=1)
    df_out.to_csv(data_dir / "output.csv", index=False)


    # === Average expression plot ===
    gene_means_true = true_inv.mean(axis=0)
    gene_means_pred = pred_inv.mean(axis=0)
    plt.figure(figsize=(8, 6))
    x = np.arange(len(gene_names))
    plt.bar(x - 0.2, gene_means_true, 0.4, label='Original')
    plt.bar(x + 0.2, gene_means_pred, 0.4, label='Predicted')
    plt.xticks(x, gene_names, rotation=45, ha='right')
    plt.ylabel("Mean Expression")
    plt.title("Average Expression per Gene")
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_dir / "avg_gene_expression_barplot.png", dpi=150)
    plt.show()

    #!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

def main():
    BASE_DIR = Path(__file__).parent.resolve()
    data_dir = BASE_DIR / "data"
    excel_path = data_dir / "NP-PC Database(Part).xlsx"

    df = pd.read_excel(excel_path, engine="openpyxl")
    df.columns = df.columns.str.strip()

    print("\nEnter condition columns (or 'done'):")
    condition_columns = []
    while True:
        c = input("  Condition column: ").strip()
        if c.lower() == "done": break
        condition_columns.append(c)

    print("\nEnter gene column letters (e.g. L, M, etc) (or 'done'):")
    gene_columns = []
    while True:
        g = input("  Gene column: ").strip()
        if g.lower() == "done": break
        gene_columns.append(g)

    def col_to_idx(name):
        idx = 0
        for ch in name.upper():
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
        return idx - 1

    gene_idx = [col_to_idx(g) for g in gene_columns]
    gene_names = df.columns[gene_idx]
    X_raw = df.iloc[:, gene_idx].values.astype(np.float32)

    # Drop rows where all genes are 0
    mask = ~(X_raw == 0).all(axis=1)
    df = df[mask].reset_index(drop=True)
    X_raw = X_raw[mask]

    df_selected = df[condition_columns].reset_index(drop=True).join(
        pd.DataFrame(X_raw, columns=gene_names))
    df_selected.to_csv(data_dir / "input.csv", index=False)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)
    expr_dim = X.shape[1]
    gene_weights_np = 1.0 / (X_raw.std(axis=0) + 1e-6)
    gene_weights = torch.tensor(gene_weights_np, dtype=torch.float32)

    encoders = {
        col: {v: [1 if i == j else 0 for j in range(len(vals))]
              for i, v in enumerate(vals)}
        for col, vals in ((col, df[col].unique().tolist()) for col in condition_columns)
    }
    C = np.array([
        sum((encoders[col][row[col]] for col in condition_columns), [])
        for _, row in df.iterrows()
    ], dtype=np.float32)
    cond_dim = C.shape[1]

    Xtr, Xv, Ctr, Cv = train_test_split(X, C, test_size=0.2, random_state=42)
    train_loader = DataLoader(GeneDataset(Xtr, Ctr), batch_size=128, shuffle=True)
    val_loader = DataLoader(GeneDataset(Xv, Cv), batch_size=128, shuffle=False)

    latent_dim = 256
    kl_weight = 0.01
    enc = Encoder(expr_dim, cond_dim, latent_dim)
    dec = Decoder(expr_dim, cond_dim, latent_dim)
    opt_e = optim.AdamW(enc.parameters(), lr=5e-4)
    opt_d = optim.AdamW(dec.parameters(), lr=5e-4)

    for epoch in range(1, 501):
        enc.train(); dec.train()
        for x_b, c_b in train_loader:
            mu, logv = enc(x_b, c_b)
            z = reparam(mu, logv)
            x_hat = dec(z, c_b)
            recon_loss = weighted_mse(x_hat, x_b, gene_weights)
            kl_loss = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()
            loss = 10 * recon_loss + kl_weight * kl_loss
            opt_e.zero_grad(); opt_d.zero_grad()
            loss.backward()
            opt_e.step(); opt_d.step()
        print(f"Epoch {epoch:03d} | Recon={recon_loss:.4f} | KL={kl_loss:.4f}")

    enc.eval(); dec.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for x_b, c_b in val_loader:
            mu, logv = enc(x_b, c_b)
            z = reparam(mu, logv)
            x_hat = dec(z, c_b)
            all_t.append(x_b.numpy())
            all_p.append(x_hat.numpy())

    true_inv = scaler.inverse_transform(np.vstack(all_t))
    pred_inv = scaler.inverse_transform(np.vstack(all_p))
    rmse = np.sqrt(mean_squared_error(true_inv, pred_inv))
    r2 = r2_score(true_inv, pred_inv)
    scaled_rmse = np.sqrt(((np.vstack(all_t) - np.vstack(all_p))**2).mean())
    print(f"\nVal RMSE: {rmse:.4f}, R²: {r2:.4f}, Scaled RMSE: {scaled_rmse:.4f}\n")

    # === Save predictions to output.csv ===
    with torch.no_grad():
        generated = []
        for c_vec in torch.tensor(C):
            z = torch.randn(1, latent_dim)
            x_pred = dec(z, c_vec.unsqueeze(0)).squeeze(0).numpy()
            generated.append(x_pred)
    df_pred = pd.DataFrame(scaler.inverse_transform(generated), columns=gene_names)
    df_out = pd.concat([df[condition_columns].reset_index(drop=True), df_pred], axis=1)
    df_out.to_csv(data_dir / "output.csv", index=False)
    print("✅ Saved predictions to output.csv")

    # === Average expression plot ===
    gene_means_true = true_inv.mean(axis=0)
    gene_means_pred = pred_inv.mean(axis=0)
    plt.figure(figsize=(8, 6))
    x = np.arange(len(gene_names))
    plt.bar(x - 0.2, gene_means_true, 0.4, label='Original')
    plt.bar(x + 0.2, gene_means_pred, 0.4, label='Predicted')
    plt.xticks(x, gene_names, rotation=45, ha='right')
    plt.ylabel("Mean Expression")
    plt.title("Average Expression per Gene")
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_dir / "avg_gene_expression_barplot.png", dpi=150)
    plt.show()



if __name__ == "__main__":
    main()
