#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import re
from sklearn.manifold import TSNE

from components import GeneDataset, Encoder, DecoderBig as Decoder, reparam, combined_loss, kl_anneal

def excel_col_letter_to_index(col_letter):
    col_letter = col_letter.upper()
    idx = 0
    for i, char in enumerate(reversed(col_letter)):
        idx += (ord(char) - ord('A') + 1) * (26 ** i)
    return idx - 1

def plot_pca_tsne(real_csv, synthetic_csv, gene_names, save_dir):
    df_real = pd.read_csv(real_csv)
    df_fake = pd.read_csv(synthetic_csv)

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

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000)
    X_tsne = tsne.fit_transform(combined)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, alpha=0.6)
    plt.title("t-SNE: Real vs. Synthetic")
    plt.tight_layout()
    plt.savefig(save_dir / "tsne_real_vs_synthetic.png")
    plt.close()

    print("Saved PCA and t-SNE plots comparing real vs. synthetic data.")

def main():
    # === Setup ===
    BASE_DIR = Path(__file__).parent.resolve()
    data_dir = BASE_DIR / "data"
    plot_dir = BASE_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)
    excel_path = data_dir / "test.xlsx"

    df = pd.read_excel(excel_path, engine="openpyxl")
    df.columns = df.columns.str.strip()

    # === Step 1: Get condition columns ===
    print("\nEnter Condition Column (or 'done', 'all'):")
    condition_columns = []
    condition_values_dict = {}

    col_input = input("  Condition column: ").strip()
    if col_input.lower() == "all":
        start_col = input("  Enter starting condition column: ").strip()
        end_col = input("  Enter ending condition column: ").strip()
        if start_col not in df.columns or end_col not in df.columns:
            raise ValueError("  One or both condition columns not found.")
        start_idx = df.columns.get_loc(start_col)
        end_idx = df.columns.get_loc(end_col)
        if end_idx < start_idx:
            raise ValueError("  Ending condition column comes before start.")
        condition_columns = list(df.columns[start_idx:end_idx + 1])
        for col in condition_columns:
            condition_values_dict[col] = df[col].dropna().unique().tolist()
    else:
        while col_input.lower() != "done":
            if col_input not in df.columns:
                print(f"  Column '{col_input}' not found. Try again.")
            else:
                condition_columns.append(col_input)
                unique_vals = df[col_input].dropna().unique()
                print(f"    Values for '{col_input}': {list(unique_vals)}")
                val_input = input("    Select values (comma-separated) or type 'all': ").strip()
                if val_input.lower() == "all":
                    condition_values_dict[col_input] = list(unique_vals)
                else:
                    condition_values_dict[col_input] = [v.strip() for v in val_input.split(",") if v.strip() in unique_vals]
            col_input = input("  Condition column: ").strip()

    # === Step 2: Species filter ===
    species_col = input("\nEnter Species Column Name: ").strip()
    if species_col and species_col in df.columns:
        unique_species = df[species_col].dropna().unique()
        print(f"  Species: {list(unique_species)}")
        species_input = input("  Species to use (or 'done', 'all'): ").strip()
        selected_species = list(unique_species) if species_input.lower() == "all" else [s.strip() for s in species_input.split(",") if s.strip() in unique_species]
    else:
        print("  No species column provided or column not found.")
        species_col = None
        selected_species = None

    # === Step 3: Total RPA filter ===
    rpa_col_input = input("\nEnter column name for Total RPA (or press Enter to skip): ").strip()
    if rpa_col_input:
        try:
            rpa_col = rpa_col_input if rpa_col_input in df.columns else df.columns[excel_col_letter_to_index(rpa_col_input)]
            min_rpa = float(input("  Enter minimum Total RPA value to include: ").strip())
        except:
            raise ValueError("Invalid RPA column or minimum value.")
    else:
        rpa_col = None
        min_rpa = None

    # === Step 4: Apply filters ===
    mask = pd.Series(True, index=df.index)
    for col in condition_columns:
        mask &= df[col].isin(condition_values_dict[col])
    if species_col and selected_species:
        mask &= df[species_col].isin(selected_species)
    if rpa_col and min_rpa is not None:
        mask &= df[rpa_col].astype(float) >= min_rpa
        print(f"Filtered rows with {rpa_col} ≥ {min_rpa}. Remaining rows: {mask.sum()}")

    df = df[mask].reset_index(drop=True)

    # === Step 5: Gene column selection ===
    print("\nEnter starting gene column (name like 'Complement C3' or letter like 'M'):")
    while True:
        start_col_input = input("  Starting gene column: ").strip()
        if start_col_input in df.columns:
            start_gene_idx = df.columns.get_loc(start_col_input)
            break
        try:
            start_gene_idx = excel_col_letter_to_index(start_col_input)
            if 0 <= start_gene_idx < len(df.columns):
                break
        except:
            pass
        print(f"  Column '{start_col_input}' not found. Try again.")

    print("\nEnter gene column names (e.g. 'Complement C3', 'Clusterin') or type 'all' to use all columns starting from the starting gene column.")
    gene_input = input("  Column name: ").strip()

    if gene_input.lower() == "all":
        end_col = input("  Enter ending gene column: ").strip()
        try:
            end_gene_idx = df.columns.get_loc(end_col) if end_col in df.columns else excel_col_letter_to_index(end_col)
            if end_gene_idx < start_gene_idx:
                raise ValueError
        except:
            raise ValueError("  Ending column not found or invalid.")
        gene_names = list(df.columns[start_gene_idx:end_gene_idx + 1])
    else:
        gene_names = [g.strip() for g in gene_input.split(",") if g.strip() in df.columns]

    # === Data Preparation ===
    X_raw = df[gene_names].values.astype(np.float32)
    print(f"Selected {len(gene_names)} gene columns. Matrix size: {X_raw.shape}")

    df_selected = df[condition_columns].reset_index(drop=True).join(pd.DataFrame(X_raw, columns=gene_names))
    df_selected.to_csv(data_dir / "input.csv", index=False)

    encoders = {
        col: {v: [1 if i == j else 0 for j, _ in enumerate(vals)] for i, v in enumerate(vals)}
        for col, vals in ((col, df[col].unique().tolist()) for col in condition_columns)
    }
    C = np.array([sum((encoders[col][row[col]] for col in condition_columns), []) for _, row in df.iterrows()], dtype=np.float32)
    cond_dim = C.shape[1]

    Xtr_raw, Xv_raw, Ctr, Cv = train_test_split(X_raw, C, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(Xtr_raw)
    Xv = scaler.transform(Xv_raw)

    expr_dim = Xtr.shape[1]
    gene_weights_np = 1.0 / (Xtr_raw.std(axis=0) + 1e-6)
    gene_weights_np /= gene_weights_np.mean()
    gene_weights = torch.tensor(gene_weights_np, dtype=torch.float32)

    train_loader = DataLoader(GeneDataset(Xtr, Ctr), batch_size=64, shuffle=True)
    val_loader = DataLoader(GeneDataset(Xv, Cv), batch_size=128, shuffle=False)

    # === Model ===
    latent_dim = 512
    enc = Encoder(expr_dim, cond_dim, latent_dim)
    dec = Decoder(expr_dim, cond_dim, latent_dim)
    opt_e = optim.Adam(enc.parameters(), lr=1e-4)
    opt_d = optim.Adam(dec.parameters(), lr=1e-4)
    scheduler_e = optim.lr_scheduler.StepLR(opt_e, step_size=200, gamma=0.9)
    scheduler_d = optim.lr_scheduler.StepLR(opt_d, step_size=200, gamma=0.9)

    # === Training ===
    for epoch in range(1, 1201):
        enc.train(); dec.train()
        kl_weight = kl_anneal(epoch, warmup_epochs=50)
        for x_b, c_b in train_loader:
            mu, logv = enc(x_b, c_b)
            z = reparam(mu, logv)
            x_hat = dec(z, c_b)
            recon_loss = combined_loss(x_hat, x_b, gene_weights)
            kl_loss = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()
            kl_weight = min(1.0, epoch/300) * 0.01
            loss = recon_loss + kl_weight * kl_loss
            opt_e.zero_grad(); opt_d.zero_grad()
            loss.backward()
            opt_e.step(); opt_d.step()
        scheduler_e.step()
        scheduler_d.step()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=1.0)

    # === Evaluation ===
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

    for i, gene in enumerate(gene_names):
        true_vals = true_inv[:, i]
        pred_vals = pred_inv[:, i]
        plt.figure(figsize=(6, 4))
        sns.histplot(true_vals, label='True', color='blue', stat='density', bins=30, kde=True, alpha=0.5)
        sns.histplot(pred_vals, label='Predicted', color='orange', stat='density', bins=30, kde=True, alpha=0.5)
        plt.title(f"{gene} – True vs Predicted")
        plt.xlabel("Gene Expression")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        safe_gene_name = re.sub(r'[\\/*?:"<>|]', "_", gene)
        plt.savefig(plot_dir / f"{safe_gene_name.replace(' ', '_')}_hist.png")
        plt.close()

    # === Metrics ===
    print(f"Saved per-gene histograms to: {plot_dir}")
    rmse = np.sqrt(mean_squared_error(true_inv, pred_inv))
    r2 = r2_score(true_inv, pred_inv)

    # === Training Metrics ===
    train_t, train_p = [], []
    with torch.no_grad():
        for x_b, c_b in DataLoader(GeneDataset(Xtr, Ctr), batch_size=128, shuffle=False):
            mu, logv = enc(x_b, c_b)
            z = reparam(mu, logv)
            x_hat = dec(z, c_b)
            train_t.append(x_b.numpy())
            train_p.append(x_hat.numpy())

    train_true = scaler.inverse_transform(np.vstack(train_t))
    train_pred = scaler.inverse_transform(np.vstack(train_p))
    train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
    print(f"Training RMSE: {train_rmse:.4f}")

    train_gene_rmse_scores = []
    for i, gene in enumerate(gene_names):
        rmse = np.sqrt(mean_squared_error(train_true[:, i], train_pred[:, i]))
        train_gene_rmse_scores.append({"Gene": gene, "Train_RMSE": rmse})

    train_rmse_df = pd.DataFrame(train_gene_rmse_scores)
    train_rmse_path = data_dir / "train_gene_rmse_scores.csv"
    train_rmse_df.to_csv(train_rmse_path, index=False)
    print(f"Saved per-gene Training RMSEs to: {train_rmse_path}")

    gene_r2_scores = []
    total_r2 = 0
    for i, gene in enumerate(gene_names):
        r2 = r2_score(true_inv[:, i], pred_inv[:, i])
        total_r2 += r2
        gene_r2_scores.append({"Gene": gene, "R2": r2})

    r2_df = pd.DataFrame(gene_r2_scores)
    r2_csv_path = data_dir / "gene_r2_scores.csv"
    r2_df.to_csv(r2_csv_path, index=False)
    print(f"\nSaved per-gene R² values to: {r2_csv_path}")

    scaled_rmse = np.sqrt(((np.vstack(all_t) - np.vstack(all_p))**2).mean())
    print(f"\nVal RMSE: {rmse:.4f}, Average R²: {(total_r2/len(gene_names)):.4f}, Scaled RMSE: {scaled_rmse:.4f}\n")

    # === Generate full synthetic predictions ===
    with torch.no_grad():
        generated = []
        for c_vec in torch.tensor(C):
            z = torch.randn(1, latent_dim)
            x_pred = dec(z, c_vec.unsqueeze(0)).squeeze(0).numpy()
            generated.append(x_pred)

    df_pred = pd.DataFrame(scaler.inverse_transform(generated), columns=gene_names)
    df_out = pd.concat([df[condition_columns].reset_index(drop=True), df_pred], axis=1)
    df_out.to_csv(data_dir / "output.csv", index=False)

    plot_pca_tsne(
        real_csv=data_dir / "input.csv",
        synthetic_csv=data_dir / "output.csv",
        gene_names=gene_names,
        save_dir=data_dir
    )

if __name__ == "__main__":
    main()
