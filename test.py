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
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

    # Step 1: Condition inputs
    print("\nEnter Condition Columns (or 'done'):")
    condition_columns = []
    condition_values_dict = {}

    while True:
        col = input("  Condition column: ").strip()
        if col.lower() == "done": break
        if col not in df.columns:
            print(f"  Column '{col}' not found. Try again.")
            continue

        condition_columns.append(col)
        unique_vals = df[col].dropna().unique()
        print(f"    Values for '{col}': {list(unique_vals)}")
        val_input = input("    Condition values (comma-separated or 'all'): ").strip()
        if val_input.lower() == "all":
            condition_values_dict[col] = list(unique_vals)
        else:
            selected_vals = [v.strip() for v in val_input.split(",") if v.strip() in unique_vals]
            condition_values_dict[col] = selected_vals

    # Step 2: Optional species filter
    species_col = input("  Species Column Name: ").strip()
    if species_col and species_col in df.columns:
        unique_species = df[species_col].dropna().unique()
        print(f"    Species: {list(unique_species)}")
        species_input = input("    Species values to use (comma-separated or 'all'): ").strip()
        if species_input.lower() == "all":
            selected_species = list(unique_species)
        else:
            selected_species = [s.strip() for s in species_input.split(",") if s.strip() in unique_species]
    else:
        print("  No species column provided or column not found.")
        species_col = None
        selected_species = None

    # Step 3: Filtering
    mask = pd.Series(True, index=df.index)
    for col in condition_columns:
        mask &= df[col].isin(condition_values_dict[col])
    if species_col and selected_species:
        mask &= df[species_col].isin(selected_species)
    df = df[mask].reset_index(drop=True)

    # Step 4: Gene column input
    print("\nEnter gene column letters (e.g. L, M, etc) (or 'done'):")
    gene_columns = []
    column_names = list(df.columns)
    while True:
        g = input("  Gene column: ").strip()
        if g.lower() == "done": break
        try:
            idx = sum((ord(ch) - ord('A') + 1) * (26 ** i) for i, ch in enumerate(reversed(g.upper()))) - 1
            if idx < 0 or idx >= len(column_names):
                raise ValueError
            gene_columns.append(g)
        except:
            print(f"  Column letter '{g}' is invalid or out of range. Try again.")

    def col_to_idx(name):
        idx = 0
        for ch in name.upper():
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
        return idx - 1

    gene_idx = [col_to_idx(g) for g in gene_columns]
    gene_names = df.columns[gene_idx]
    X_raw = df.iloc[:, gene_idx].values.astype(np.float32)

    df_selected = df[condition_columns].reset_index(drop=True).join(
        pd.DataFrame(X_raw, columns=gene_names))
    df_selected.to_csv(data_dir / "input.csv", index=False)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_raw)
    expr_dim = X.shape[1]
    gene_weights_np = 1.0 / (X_raw.std(axis=0) + 1e-6)
    gene_weights = torch.tensor(gene_weights_np, dtype=torch.float32)

    encoders = {
        col: {v: [1 if i == j else 0 for j, _ in enumerate(vals)]
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
    # === Generate full predictions on entire dataset (not just validation)
    with torch.no_grad():
        generated = []
        for c_vec in torch.tensor(C):
            z = torch.randn(1, latent_dim)
            x_pred = dec(z, c_vec.unsqueeze(0)).squeeze(0).numpy()
            generated.append(x_pred)

    df_pred = pd.DataFrame(scaler.inverse_transform(generated), columns=gene_names)
    df_out = pd.concat([df[condition_columns].reset_index(drop=True), df_pred], axis=1)
    df_out.to_csv(data_dir / "output.csv", index=False)

    generate_synthetic_samples(
        decoder=dec,
        condition_df=df[condition_columns],
        encoder_dict=encoders,
        gene_names=gene_names,
        latent_dim=latent_dim,
        scaler=scaler,
        output_path=data_dir / "synthetic_output.csv",
        data_dir=data_dir,
        n_samples=100
    )

    plot_pca_tsne(
        real_csv=data_dir / "output.csv",
        synthetic_csv=data_dir / "synthetic_output.csv",
        gene_names=gene_names,
        save_dir=data_dir
    )

def generate_synthetic_samples(decoder, condition_df, encoder_dict, gene_names, latent_dim, scaler, output_path, data_dir, n_samples=100):
    decoder.eval()
    synthetic_data = []
    cond_data = []
    for i in range(n_samples):
        row = condition_df.sample(1).iloc[0]
        cond_vector = sum((encoder_dict[col][row[col]] for col in encoder_dict), [])
        cond_tensor = torch.tensor([cond_vector], dtype=torch.float32)
        z = torch.randn(1, latent_dim)
        with torch.no_grad():
            x_pred = decoder(z, cond_tensor).squeeze(0).numpy()
            x_pred_inv = scaler.inverse_transform([x_pred])[0]
        synthetic_data.append(x_pred_inv)
        cond_data.append(row[encoder_dict.keys()].values.tolist())
    cond_df = pd.DataFrame(cond_data, columns=encoder_dict.keys())
    expr_df = pd.DataFrame(synthetic_data, columns=gene_names)
    df_out = pd.concat([cond_df, expr_df], axis=1)
    df_out.to_csv(output_path, index=False)
    print(f"Saved {n_samples} synthetic samples to {output_path}")

    df_real = pd.read_csv(data_dir / "output.csv")
    df_fake = pd.read_csv(output_path)
    for gene in gene_names:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(df_real[gene], label='Real', fill=True)
        sns.kdeplot(df_fake[gene], label='Synthetic', fill=True)
        plt.title(f'Distribution Comparison - {gene}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(data_dir / f"{gene}_distribution_comparison.png")
        plt.close()

def plot_pca_tsne(real_csv, synthetic_csv, gene_names, save_dir):
    df_real = pd.read_csv(real_csv)
    df_fake = pd.read_csv(synthetic_csv)

    # Ensure enough samples and features
    if len(df_real) + len(df_fake) < 3 or len(gene_names) < 2:
        print("Skipping PCA/t-SNE: Not enough samples or genes to reduce.")
        return

    X_real = df_real[gene_names].values
    X_fake = df_fake[gene_names].values
    combined = np.vstack([X_real, X_fake])
    labels = ['Real'] * len(X_real) + ['Synthetic'] * len(X_fake)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(combined)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, alpha=0.6)
    plt.title("PCA: Real vs. Synthetic")
    plt.tight_layout()
    plt.savefig(save_dir / "pca_real_vs_synthetic.png")
    plt.close()
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    X_tsne = tsne.fit_transform(combined)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, alpha=0.6)
    plt.title("t-SNE: Real vs. Synthetic")
    plt.tight_layout()
    plt.savefig(save_dir / "tsne_real_vs_synthetic.png")
    plt.close()
    print("Saved PCA and t-SNE plots comparing real vs. synthetic data.")

if __name__ == "__main__":
    main()
