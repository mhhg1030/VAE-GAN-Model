import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ks_2samp

from components import GeneDataset, Encoder, Decoder, reparam, combined_loss, kl_anneal
from plot import plot_histogram, plot_pca_tsne

def train_model(
    Xtr, Xv, Ctr, Cv,
    cond_names,               
    gene_names, gene_names_all,
    scaler, data_dir, plot_dir,
    latent_dim=256, 
    ks_boost=2.0
):
    # --- initialize model & weights ---
    expr_dim = Xtr.shape[1]
    cond_dim = Ctr.shape[1]
    latent_dim = 256

    enc = Encoder(expr_dim, cond_dim, latent_dim)
    dec = Decoder(expr_dim, cond_dim, latent_dim)

    # base gene weights (inverse‐std), sum to 1
    base_w = 1.0 / (Xtr.std(axis=0) + 1e-4)
    base_w /= base_w.sum()
    gene_weights = torch.tensor(base_w, dtype=torch.float32)

    # loaders
    train_loader = DataLoader(GeneDataset(Xtr, Ctr), batch_size=64, shuffle=True)
    val_loader   = DataLoader(GeneDataset(Xv, Cv), batch_size=128, shuffle=False)

    # optimizers & schedulers
    opt_e = torch.optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-6)
    opt_d = torch.optim.Adam(dec.parameters(), lr=1e-3, weight_decay=1e-6)
    sched_e = StepLR(opt_e, step_size=200, gamma=0.7)
    sched_d = StepLR(opt_d, step_size=200, gamma=0.7)

    # --- training loop ---
    num_epochs = 1500
    for epoch in range(1, num_epochs + 1):
        enc.train(); dec.train()
        kl_weight = kl_anneal(epoch, warmup_epochs=100)
        for x_b, c_b in train_loader:
            mu, logv = enc(x_b, c_b)
            z = reparam(mu, logv)
            x_hat = dec(z, c_b)

            recon = combined_loss(x_hat, x_b, gene_weights)
            kl    = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()
            loss  = recon + kl_weight * kl

            opt_e.zero_grad(); opt_d.zero_grad()
            loss.backward()
            opt_e.step(); opt_d.step()

        sched_e.step(); sched_d.step()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)

    enc.eval(); dec.eval()

    # helper to collect and inverse‐scale batches
    def collect(loader):
        trues, preds = [], []
        with torch.no_grad():
            for xb, cb in loader:
                mu, logv = enc(xb, cb)
                z = reparam(mu, logv)
                xh = dec(z, cb)
                trues.append(xb.cpu().numpy())
                preds.append(xh.cpu().numpy())
        true_full = scaler.inverse_transform(np.vstack(trues))
        pred_full = scaler.inverse_transform(np.vstack(preds))
        return true_full, pred_full

    # get full train & val predictions
    train_true_full, train_pred_full = collect(DataLoader(GeneDataset(Xtr, Ctr), batch_size=128, shuffle=False))
    val_true_full,   val_pred_full   = collect(val_loader)

    # --- per‐gene RMSE on training, filter out RMSE>1 ---
    all_rmse, ignored, kept = [], [], []
    for idx, gene in enumerate(gene_names_all):
        rmse = np.sqrt(mean_squared_error(train_true_full[:, idx], train_pred_full[:, idx]))
        entry = {"Gene": gene, "Train_RMSE": rmse}
        all_rmse.append(entry)
        (ignored if rmse > 1.0 else kept).append(entry)

    pd.DataFrame(all_rmse).to_csv(data_dir/"train_gene_rmse_scores.csv", index=False)
    pd.DataFrame(ignored).to_csv(data_dir/"ignored_high_rmse_genes.csv",  index=False)

    gene_names[:] = [e["Gene"] for e in kept]
    kept_idx = [i for i,g in enumerate(gene_names_all) if g in gene_names]

    # print train RMSE summary
    full_rmse = np.sqrt(mean_squared_error(train_true_full, train_pred_full))
    filt_rmse = np.sqrt(mean_squared_error(
        train_true_full[:, kept_idx],
        train_pred_full[:, kept_idx]
    ))
    print(f"\nFull Training RMSE (all genes): {full_rmse:.4f}")
    print(f"Filtered Training RMSE: {filt_rmse:.4f}\n")

    # --- validation RMSE ---
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
    pd.DataFrame(ks_results).to_csv(data_dir/"ks_test_results.csv", index=False)
    offenders = {r["Gene"] for r in ks_results if r["p_value"] <= 0.05}
    # # Sanity‐check one gene
    # i = 0
    # rv = val_true_full[:, kept_idx[i]]
    # sv = val_pred_full[:, kept_idx[i]]
    # print("Example gene:", gene_names[0])
    # print("  real_vals:", rv[:5], "… shape:", rv.shape)
    # print("  synth_vals:", sv[:5], "… shape:", sv.shape)

    # stat, pval = ks_2samp(rv, sv)
    # print("  KS_stat, pval:", stat, pval)
    # --- upweight KS offenders and rebuild gene_weights ---
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
            train_true_full[:, kept_idx][..., i],
            train_pred_full[:, kept_idx][..., i]
        )
        train_r2.append({"Gene": gene, "R2": r2v})
    pd.DataFrame(train_r2).to_csv(data_dir/"gene_r2_scores.csv", index=False)

    # Validation R²
    val_r2 = []
    for i, gene in enumerate(gene_names):
        r2v = r2_score(val_true[:, i], val_pred[:, i])
        val_r2.append({"Gene": gene, "R2": r2v})
    pd.DataFrame(val_r2).to_csv(data_dir/"validation_gene_r2_scores.csv", index=False)

    # print R² averages cleanly
    train_avg = pd.read_csv(data_dir/"gene_r2_scores.csv")["R2"].mean()
    val_avg   = pd.read_csv(data_dir/"validation_gene_r2_scores.csv")["R2"].mean()
    print(f"Training R² Avg: {train_avg:.4f}")
    print(f"Validation R² Avg: {val_avg:.4f}\n")

    # --- generate synthetic output.csv with condition names ---
    # rep_output = 3
    # all_cond
    
    synth = []
    with torch.no_grad():
        for c_vec in torch.tensor(np.vstack([Ctr, Cv])):
            z = torch.randn(1, latent_dim)
            pred = dec(z, c_vec.unsqueeze(0)).squeeze(0).cpu().numpy()
            synth.append(pred)

    synth = np.clip(np.vstack(synth), 0, None)
    synth_inv = scaler.inverse_transform(synth)[:, kept_idx]

    input_df = pd.read_csv(data_dir / "input.csv")
    cond_df  = input_df.drop(columns=gene_names) 
    df_pred = pd.DataFrame(synth_inv, columns=gene_names)
    df_out  = pd.concat([cond_df, df_pred], axis=1)
    df_out.to_csv(data_dir/"output.csv", index=False)

    # PCA & t-SNE
    plot_pca_tsne(
        real_csv=data_dir/"input.csv",
        synthetic_csv=data_dir/"output.csv",
        gene_names=gene_names,
        save_dir=data_dir
    )

    return enc, dec, gene_names
