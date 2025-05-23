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
from sklearn.preprocessing import StandardScaler
def main():
    BASE_DIR = Path(__file__).parent.resolve()
    data_dir = BASE_DIR / "data"
    plot_dir  = BASE_DIR /"plots"
    plot_dir.mkdir(exist_ok=True)
    
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

        # condition_columns.append(col)
        # unique_vals = df[col].dropna().unique()
        # print(f"    Values for '{col}': {list(unique_vals)}")
        # val_input = input("    Condition values (comma-separated or 'all'): ").strip()
        # if val_input.lower() == "all":
        #     condition_values_dict[col] = list(unique_vals)
        # else:
        #     selected_vals = [v.strip() for v in val_input.split(",") if v.strip() in unique_vals]
        #     condition_values_dict[col] = selected_vals

    # Step 2: Optional species filter
    species_col = input("\n  Species Column Name: ").strip()
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
    
    def excel_col_letter_to_index(col_letter):
        col_letter = col_letter.upper()
        idx = 0
        for i, char in enumerate(reversed(col_letter)):
            idx += (ord(char) - ord('A') + 1) * (26 ** i)
        return idx - 1

    # === Prompt 1: Ask user for starting gene column by name or letter
    while True:
        start_col = input("\nEnter starting gene column (name like 'Complement C3' or letter like 'L'): ").strip()
        if start_col in df.columns:
            start_idx = df.columns.get_loc(start_col)
            break
        else:
            try:
                idx = excel_col_letter_to_index(start_col)
                if idx >= len(df.columns):
                    raise ValueError
                start_idx = idx
                break
            except:
                print(f"  Column '{start_col}' not found. Try again.")

    # === Prompt 2: Ask whether to use all or individual gene columns
    print(f"\nEnter gene column names (e.g. 'Complement C3', 'Clusterin') or type 'all' to use all columns starting from '{df.columns[start_idx]}'.")
    gene_names = ['Complement C3', 'Vitronectin', 'Immunoglobulin heavy constant mu', 'Apolipoprotein A-II', 'Immunoglobulin heavy constant gamma 1', 'Immunoglobulin kappa constant', 'Prothrombin','Apolipoprotein B-100', 'Complement C4-B', 
                  'Alpha-1-antitrypsin', 'Gelsolin']
    # while True:
    #     g = input("  Gene column: ").strip()
    #     if g.lower() == "done":
    #         break
    #     elif g.lower() == "all":
    #         candidate_cols = df.columns[start_idx:]
    #         gene_names = [col for col in candidate_cols if pd.api.types.is_numeric_dtype(df[col])]
    #         break
    #     else:
    #         if g not in df.columns:
    #             print(f"  Column '{g}' not found. Try again.")
    #         elif not pd.api.types.is_numeric_dtype(df[g]):
    #             print(f"  Column '{g}' is not numeric and cannot be used.")
    #         else:
    #             gene_names.append(g)

    # # === Final check to avoid MinMaxScaler crash
    # if not gene_names:
    #     print("No valid gene columns selected. Exiting.")
    #     exit()

    # === Extract expression matrix
    X_raw = df[gene_names].values.astype(np.float32)
    print(f"Selected {len(gene_names)} gene columns. Matrix size: {X_raw.shape}")

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

    # replace X_small/C_small block with:
    Xtr, Xv, Ctr, Cv = train_test_split(X, C, test_size=0.2, random_state=42)
    train_loader = DataLoader(GeneDataset(Xtr, Ctr), batch_size=128, shuffle=True)
    val_loader   = DataLoader(GeneDataset(Xv, Cv), batch_size=128, shuffle=False)


    latent_dim = 512  # updated for more capacity
    enc = Encoder(expr_dim, cond_dim, latent_dim)
    dec = Decoder(expr_dim, cond_dim, latent_dim)
    opt_e = optim.Adam(enc.parameters(), lr=1e-4)
    opt_d = optim.Adam(dec.parameters(), lr=1e-4)

    scheduler_e = optim.lr_scheduler.StepLR(opt_e, step_size=200, gamma=0.9)
    scheduler_d = optim.lr_scheduler.StepLR(opt_d, step_size=200, gamma=0.9)

    for epoch in range(1, 600):
        enc.train(); dec.train()
        kl_weight = kl_anneal(epoch, warmup_epochs=50)
        for x_b, c_b in train_loader:
            mu, logv = enc(x_b, c_b)
            z = reparam(mu, logv)
            x_hat = dec(z, c_b)

            recon_loss = combined_loss(x_hat, x_b, gene_weights)
            kl_loss = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()
            kl_weight = min(1.0, epoch/300) * 0.01    # cap at 0.05
            loss = recon_loss + kl_weight * kl_loss



            opt_e.zero_grad(); opt_d.zero_grad()
            loss.backward()
            #checking if gradients are exploding/vanishing 
            total_norm = 0
            for p in enc.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item()**2
            total_norm = total_norm**0.5
            print(f"[Encoder] Grad norm: {total_norm:.4f}")
            opt_e.step(); opt_d.step()

        scheduler_e.step()
        scheduler_d.step()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), max_norm=1.0)


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
    

    # Evaluate per-gene R² and save to CSV
    gene_r2_scores = []

    for i, gene in enumerate(gene_names):
        r2 = r2_score(true_inv[:, i], pred_inv[:, i])
        gene_r2_scores.append({"Gene": gene, "R2": r2})
        # print(f"{gene}: R² = {r2:.4f}")

    # Convert to DataFrame and save
    r2_df = pd.DataFrame(gene_r2_scores)
    r2_csv_path = data_dir / "gene_r2_scores.csv"
    r2_df.to_csv(r2_csv_path, index=False)
    print(f"\nSaved per-gene R² values to: {r2_csv_path}")

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
        plot_dir=plot_dir
    )


    plot_pca_tsne(
        real_csv=data_dir / "output.csv",
        synthetic_csv=data_dir / "synthetic_output.csv",
        gene_names=gene_names,
        save_dir=data_dir
    )

def generate_synthetic_samples(decoder, condition_df, encoder_dict, gene_names,
                               latent_dim, scaler, output_path, data_dir,n_samples=100, plot_dir=None):

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

    df_real = pd.read_csv(data_dir / "input.csv")
    df_fake = pd.read_csv(data_dir / "output.csv")
    for gene in gene_names:
    # load real & fake
        real = df_real[gene].values
        fake = df_fake[gene].values

        # 1) Plot histogram of counts instead of density:
        plt.figure(figsize=(6,4))
        sns.histplot(real, label='Real', stat='count', bins=30, alpha=0.4)
        sns.histplot(fake, label='Synthetic', stat='count', bins=30, alpha=0.4)
        plt.title(f'Histogram Counts – {gene}')
        plt.legend()
        plt.savefig(plot_dir/f'{gene}_hist_counts.png')
        plt.close()

        # # 2) Plot KDE but _not_ normalizing each to area=1 (so you see absolute density):
        # plt.figure(figsize=(6,4))
        # sns.kdeplot(real, label='Real', fill=True, common_norm=False)
        # sns.kdeplot(fake, label='Synthetic', fill=True, common_norm=False)
        # plt.title(f'KDE with common_norm=False – {gene}')
        # plt.legend()
        # plt.savefig(plot_dir/f'{gene}_kde_commonnorm_false.png')
        # plt.close()


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
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000)
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
