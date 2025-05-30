import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from components import GeneDataset, Encoder, Decoder, reparam, combined_loss, kl_anneal
from plot import plot_histogram, plot_pca_tsne

def train_model(Xtr, Xv, Ctr, Cv, gene_names, gene_names_all, scaler, data_dir, plot_dir):
    expr_dim = Xtr.shape[1]
    cond_dim = Ctr.shape[1]
    latent_dim = 256

    enc = Encoder(expr_dim, cond_dim, latent_dim)
    dec = Decoder(expr_dim, cond_dim, latent_dim)

    gene_weights_np = 1.0 / (Xtr.std(axis=0) + 1e-4)
    gene_weights_np /= gene_weights_np.mean()
    gene_weights = torch.tensor(gene_weights_np, dtype=torch.float32)

    train_loader = DataLoader(GeneDataset(Xtr, Ctr), batch_size=64, shuffle=True)
    val_loader = DataLoader(GeneDataset(Xv, Cv), batch_size=128, shuffle=False)

    opt_e = torch.optim.Adam(enc.parameters(), lr=1e-4, weight_decay=1e-5)
    opt_d = torch.optim.Adam(dec.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler_e = torch.optim.lr_scheduler.StepLR(opt_e, step_size=200, gamma=1.0)
    scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=200, gamma=1.0)

    for epoch in range(1, 1201):
        enc.train(); dec.train()
        kl_weight = kl_anneal(epoch, warmup_epochs=100)
        for x_b, c_b in train_loader:
            mu, logv = enc(x_b, c_b)
            z = reparam(mu, logv)
            x_hat = dec(z, c_b)
            recon_loss = combined_loss(x_hat, x_b, gene_weights)
            kl_loss = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()
            kl_weight = min(1.0, epoch / 300) * 0.0001
            loss = recon_loss + kl_weight * kl_loss
            opt_e.zero_grad(); opt_d.zero_grad()
            loss.backward()
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

    for i, gene in enumerate(gene_names):
        plot_histogram(true_inv[:, i], pred_inv[:, i], gene, plot_dir)

    print(f"Saved per-gene histograms to: {plot_dir}")

    r2 = np.mean(r2_score(true_inv[:, i], pred_inv[:, i]))

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

    # Per-gene Training RMSE Calculation 
    train_gene_rmse_scores = []
    ignored_entries = []
    kept_entries = []

    for i, gene in enumerate(gene_names_all):
        rmse = np.sqrt(mean_squared_error(train_true[:, i], train_pred[:, i]))
        entry = {"Gene": gene, "Train_RMSE": rmse}
        if rmse > 1.0:
            ignored_entries.append(entry)
        else:
            kept_entries.append(entry)
        train_gene_rmse_scores.append(entry)

    pd.DataFrame(train_gene_rmse_scores).to_csv(data_dir / "train_gene_rmse_scores.csv", index=False)

    ignored_path = data_dir / "ignored_high_rmse_genes.csv"
    pd.DataFrame(ignored_entries).to_csv(ignored_path, index=False)
    print(f"Ignored genes (RMSE > 1) saved to: {ignored_path}")

    gene_names[:] = [entry["Gene"] for entry in kept_entries]

    train_rmse_df = pd.DataFrame(kept_entries)
    print(f"\nTotal genes evaluated: {len(train_gene_rmse_scores)}")
    print(f"Genes ignored (RMSE > 1): {len(ignored_entries)}")
    print(f"Remaining genes used: {len(gene_names)}")

    train_rmse_full = np.sqrt(mean_squared_error(train_true, train_pred))
    print(f"\nFull Training RMSE (all genes): {train_rmse_full:.4f}")

    kept_indices = [i for i, gene in enumerate(gene_names_all) if gene in gene_names]
    train_true_filtered = train_true[:, kept_indices]
    train_pred_filtered = train_pred[:, kept_indices]
    train_rmse_filtered = np.sqrt(mean_squared_error(train_true_filtered, train_pred_filtered))
    print(f"Filtered Training RMSE (ignored genes removed): {train_rmse_filtered:.4f}\n")

    # === Compute validation R² scores ===
    val_true, val_pred = [], []

    with torch.no_grad():
        for x_b, c_b in val_loader:
            mu, logv = enc(x_b, c_b)
            z = reparam(mu, logv)
            x_hat = dec(z, c_b)
            val_true.append(x_b.numpy())
            val_pred.append(x_hat.numpy())

    val_true = scaler.inverse_transform(np.vstack(val_true))
    val_pred = scaler.inverse_transform(np.vstack(val_pred))

    val_r2_scores = []
    for i, gene in enumerate(gene_names):
        r2 = r2_score(val_true[:, i], val_pred[:, i])
        val_r2_scores.append({"Gene": gene, "R2": r2})

    # Save validation R² scores
    val_r2_df = pd.DataFrame(val_r2_scores)
    val_r2_df.to_csv(data_dir / "validation_gene_r2_scores.csv", index=False)

    train_df = pd.read_csv(data_dir / "gene_r2_scores.csv")
    val_df = pd.read_csv(data_dir / "validation_gene_r2_scores.csv")

    train_avg_r2 = train_df["R2"].mean()
    val_avg_r2 = val_df["R2"].mean()

    print(f"Training R² Average: {train_avg_r2:.4f}")
    print(f"Validation R² Average: {val_avg_r2:.4f}")

    # === Save synthetic data to output.csv
    with torch.no_grad():
        generated = []
        for c_vec in torch.tensor(Ctr.tolist() + Cv.tolist()):
            z = torch.randn(1, latent_dim)
            x_pred = dec(z, c_vec.unsqueeze(0)).squeeze(0).numpy()
            generated.append(x_pred)

    generated = np.array(generated)
    generated = np.clip(generated, a_min=0, a_max=None)

    # Inverse transform the full prediction
    generated_inv = scaler.inverse_transform(generated)

    # Filter columns corresponding to kept genes
    kept_indices = [i for i, gene in enumerate(gene_names_all) if gene in gene_names]
    generated_filtered = generated_inv[:, kept_indices]
    generated = np.clip(generated, a_min=0, a_max=None) 
    df_pred = pd.DataFrame(generated_filtered, columns=gene_names)
    df_out = pd.concat([pd.DataFrame(Ctr.tolist() + Cv.tolist()), df_pred], axis=1)
    output_path = data_dir / "output.csv"
    df_out.to_csv(output_path, index=False)

    plot_pca_tsne(
        real_csv=data_dir / "input.csv",
        synthetic_csv=data_dir / "output.csv",
        gene_names=gene_names,
        save_dir=data_dir
    )

    return enc, dec, gene_names