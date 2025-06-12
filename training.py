# training.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ks_2samp
from tqdm import tqdm

from components import GeneDataset, Encoder, Decoder, reparam, combined_loss, kl_anneal
from plot import plot_histogram, plot_pca_tsne

def train_model(
    Xtr, Xv, Ctr, Cv,
    cond_names,
    gene_names, gene_names_all,
    scaler, data_dir, plot_dir,
    stats_dir,
    latent_dim=128,
    recon_alpha=2.0,
    kl_beta=0.1,
    max_lr=3e-4,
    ks_boost=2.0
):
    # --- capture raw zero‐pattern for final masking ---
    raw_combined = np.vstack([Xtr, Xv])          # shape [n_samples, G_all]
    zero_pattern = (raw_combined == 0.0)         # bool mask

    # --- initialize ---
    expr_dim = Xtr.shape[1]
    cond_dim = Ctr.shape[1]
    num_epochs = 300

    enc = Encoder(expr_dim, cond_dim, latent_dim)
    dec = Decoder(expr_dim, cond_dim, latent_dim)

    base_w = 1.0 / (Xtr.std(axis=0) + 1e-4)
    base_w /= base_w.sum()
    gene_weights = torch.tensor(base_w, dtype=torch.float32)

    train_loader = DataLoader(GeneDataset(Xtr, Ctr), batch_size=64, shuffle=True)
    val_loader   = DataLoader(GeneDataset(Xv, Cv), batch_size=64, shuffle=False)

    opt_e = torch.optim.Adam(enc.parameters(), lr=max_lr, betas=(0.5,0.9), weight_decay=1e-7)
    opt_d = torch.optim.Adam(dec.parameters(), lr=max_lr, betas=(0.5,0.9), weight_decay=1e-7)

    sched_e = OneCycleLR(opt_e, max_lr=max_lr,
                        steps_per_epoch=len(train_loader),
                        epochs=num_epochs, pct_start=0.1,
                        anneal_strategy='cos', div_factor=25.0)
    sched_d = OneCycleLR(opt_d, max_lr=max_lr,
                        steps_per_epoch=len(train_loader),
                        epochs=num_epochs, pct_start=0.1,
                        anneal_strategy='cos', div_factor=25.0)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 50

    # --- training + early stopping ---
    for epoch in range(1, num_epochs+1):
        enc.train(); dec.train()
        kl_weight = kl_anneal(epoch, warmup_epochs=200)
        epoch_loss = 0.0

        for x_b, c_b in tqdm(train_loader, desc="Batches", leave=False):
            mu, logv = enc(x_b, c_b)
            z = reparam(mu, logv)
            x_hat = dec(z, c_b)

            recon = combined_loss(x_hat, x_b, gene_weights)
            kl = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()
            loss = recon_alpha * recon + kl_beta * kl_weight * kl

            opt_e.zero_grad(); opt_d.zero_grad()
            loss.backward()
            opt_e.step(); opt_d.step()

            sched_e.step(); sched_d.step()

            epoch_loss += loss.item() * x_b.size(0)

        avg_train = epoch_loss / len(train_loader.dataset)
        tqdm.write(f"Epoch {epoch} train loss: {avg_train:.4f}")

        # validation
        enc.eval(); dec.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, c_b in val_loader:
                mu, logv = enc(x_b, c_b)
                z = reparam(mu, logv)
                x_hat = dec(z, c_b)

                recon = combined_loss(x_hat, x_b, gene_weights)
                kl = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()
                val_loss += (recon_alpha * recon + kl_beta * kl_weight * kl).item() * x_b.size(0)

        avg_val = val_loss / len(val_loader.dataset)
        tqdm.write(f"Epoch {epoch} val   loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"No improvement in {patience} epochs—stopping early.")
                break

        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)

    # --- helper to collect & inverse‐scale ---
    def collect(loader):
        trues, preds = [], []
        with torch.no_grad():
            for xb, cb in loader:
                mu, logv = enc(xb, cb)
                z = reparam(mu, logv)
                xh = dec(z, cb)
                trues.append(xb.cpu().numpy())
                preds.append(xh.cpu().numpy())
        return (
            scaler.inverse_transform(np.vstack(trues)),
            scaler.inverse_transform(np.vstack(preds))
        )

    # --- compute train & val predictions + mask zeros in preds ---
    train_true_full, train_pred_full = collect(DataLoader(GeneDataset(Xtr, Ctr),
                                                           batch_size=128, shuffle=False))
    val_true_full,   val_pred_full   = collect(val_loader)

    train_pred_full[ train_true_full == 0.0 ] = 0.0
    val_pred_full  [ val_true_full   == 0.0 ] = 0.0

    # --- per‐gene RMSE & filtering ---
    all_rmse, ignored, kept = [], [], []
    for idx, gene in enumerate(gene_names_all):
        rmse = np.sqrt(mean_squared_error(
            train_true_full[:, idx], train_pred_full[:, idx]))
        entry = {"Gene": gene, "Train_RMSE": rmse}
        all_rmse.append(entry)
        (ignored if rmse > 1.0 else kept).append(entry)

    pd.DataFrame(all_rmse) \
      .to_csv(stats_dir/"train_gene_rmse_scores.csv", index=False)
    pd.DataFrame(ignored) \
      .to_csv(stats_dir/"ignored_high_rmse_genes.csv", index=False)

    gene_names[:] = [e["Gene"] for e in kept]
    kept_idx = [i for i,g in enumerate(gene_names_all) if g in gene_names]

    # --- print RMSE summaries ---
    full_rmse = np.sqrt(mean_squared_error(train_true_full, train_pred_full))
    filt_rmse = np.sqrt(mean_squared_error(
        train_true_full[:, kept_idx],
        train_pred_full[:, kept_idx]
    ))
    print(f"\nFull Training RMSE (all genes): {full_rmse:.4f}")
    print(f"Filtered Training RMSE: {filt_rmse:.4f}\n")

    full_val_rmse = np.sqrt(mean_squared_error(val_true_full, val_pred_full))
    filt_val_rmse = np.sqrt(mean_squared_error(
        val_true_full[:, kept_idx],
        val_pred_full[:, kept_idx]
    ))
    print(f"Full Validation RMSE: {full_val_rmse:.4f}")
    print(f"Filtered Validation RMSE: {filt_val_rmse:.4f}\n")

    # --- per‐gene KS‐test ---
    ks_results = []
    for i, gene in enumerate(gene_names):
        real_vals  = val_true_full[:, kept_idx[i]]
        synth_vals = val_pred_full[:, kept_idx[i]]
        stat, pval = ks_2samp(real_vals, synth_vals)
        ks_results.append({"Gene": gene, "KS_stat": stat, "p_value": pval})
    pd.DataFrame(ks_results).to_csv(stats_dir/"ks_test_results.csv", index=False)
    offenders = {r["Gene"] for r in ks_results if r["p_value"] <= 0.05}

    boost = ks_boost
    weights = []
    for gene, w in zip(gene_names_all, base_w):
        weights.append(w*boost if gene in offenders else w)
    gene_weights_np = np.array(weights)
    gene_weights_np /= gene_weights_np.sum()
    gene_weights = torch.tensor(gene_weights_np, dtype=torch.float32)

    # --- per‐gene histograms & R² ---
    val_true = val_true_full[:, kept_idx]
    val_pred = val_pred_full[:, kept_idx]
    for i, gene in enumerate(gene_names):
        plot_histogram(val_true[:, i], val_pred[:, i], gene, plot_dir)
    print(f"Saved histograms to: {plot_dir}")

    # Training R²
    train_r2 = []
    for i, gene in enumerate(gene_names):
        r2v = r2_score(
            train_true_full[..., i],
            train_pred_full[..., i]
        )
        train_r2.append({"Gene": gene, "R2": r2v})
    pd.DataFrame(train_r2).to_csv(stats_dir/"gene_r2_scores.csv", index=False)

    # Validation R²
    val_r2 = []
    for i, gene in enumerate(gene_names):
        r2v = r2_score(val_true[:, i], val_pred[:, i])
        val_r2.append({"Gene": gene, "R2": r2v})
    pd.DataFrame(val_r2).to_csv(stats_dir/"validation_gene_r2_scores.csv", index=False)

    # print R² averages cleanly
    train_avg = pd.read_csv(stats_dir/"gene_r2_scores.csv")["R2"].mean()
    val_avg   = pd.read_csv(stats_dir/"validation_gene_r2_scores.csv")["R2"].mean()
    print(f"Training R² Avg: {train_avg:.4f}")
    print(f"Validation R² Avg: {val_avg:.4f}\n")

    synth = []
    with torch.no_grad():
        for c_vec in torch.tensor(np.vstack([Ctr, Cv])):
            z    = torch.randn(1, latent_dim)
            pred = dec(z, c_vec.unsqueeze(0)).squeeze(0).cpu().numpy()
            synth.append(pred)

    synth = np.clip(np.vstack(synth), 0, None)
    synth_inv = scaler.inverse_transform(synth)[:, kept_idx]

    zero_pattern_kept = zero_pattern[:, kept_idx]
    synth_inv[ zero_pattern_kept ] = 0.0

    epsilon = 1e-3
    synth_inv[ np.abs(synth_inv) < epsilon ] = 0.0

    input_df = pd.read_csv(stats_dir/"real.csv")
    cond_df  = input_df.drop(columns=gene_names)
    df_pred  = pd.DataFrame(synth_inv, columns=gene_names)
    df_out   = pd.concat([cond_df, df_pred], axis=1)
    df_out.to_csv(stats_dir/"synthetic.csv", index=False)

    plot_pca_tsne(
        real_csv=stats_dir/"real.csv",
        synthetic_csv=stats_dir/"synthetic.csv",
        gene_names=gene_names,
        save_dir=stats_dir
    )

    pd.read_csv(stats_dir/"real.csv").to_excel(stats_dir/"real.xlsx", index=False)
    pd.read_csv(stats_dir/"synthetic.csv").to_excel(stats_dir/"synthetic.xlsx", index=False)

    return enc, dec, gene_names
