# training.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import torch.nn.functional as F

from components import GeneDataset, Encoder, Decoder, Discriminator, reparam, combined_loss, kl_anneal, weighted_mse_loss
from plot import plot_histogram, plot_pca_tsne

def train_model(
    Xtr, Xv, Xte,
    Ctr, Cv, Cte,
    gene_names, gene_names_all,
    scaler, plot_dir,
    stats_dir,
    latent_dim=128,
    recon_alpha=2.0,
    kl_beta=0.1,
    max_lr=3e-4,
):
    raw_combined = np.vstack([Xtr, Xv, Xte])
    zero_pattern = (raw_combined == 0.0)

    expr_dim = Xtr.shape[1]
    cond_dim = Ctr.shape[1]
    num_epochs = 350

    enc = Encoder(expr_dim, cond_dim, latent_dim)
    dec = Decoder(expr_dim, cond_dim, latent_dim)
    dis = Discriminator(expr_dim, cond_dim)

    base_w = 1.0 / (Xtr.std(axis=0) + 1e-4)
    base_w /= base_w.sum()
    gene_weights = torch.tensor(base_w, dtype=torch.float32)

    train_loader = DataLoader(GeneDataset(Xtr, Ctr), batch_size=64, shuffle=True)
    val_loader   = DataLoader(GeneDataset(Xv, Cv), batch_size=64, shuffle=False)
    test_loader  = DataLoader(GeneDataset(Xte, Cte), batch_size=64, shuffle=False)

    opt_e = torch.optim.Adam(enc.parameters(), lr=max_lr, betas=(0.5, 0.9), weight_decay=1e-7)
    opt_d = torch.optim.Adam(dec.parameters(), lr=max_lr, betas=(0.5, 0.9), weight_decay=1e-7)
    opt_dis = torch.optim.Adam(dis.parameters(), lr=max_lr)

    sched_e = OneCycleLR(opt_e, max_lr=max_lr,
                         steps_per_epoch=len(train_loader),
                         epochs=num_epochs, pct_start=0.1,
                         anneal_strategy='cos', div_factor=7.0)
    sched_d = OneCycleLR(opt_d, max_lr=max_lr,
                         steps_per_epoch=len(train_loader),
                         epochs=num_epochs, pct_start=0.1,
                         anneal_strategy='cos', div_factor=7.0)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 50

    for epoch in range(1, num_epochs+1):
        enc.train(); dec.train(); dis.train()
        kl_weight = kl_anneal(epoch, warmup_epochs=200)
        epoch_loss = 0.0
        adv_alpha = 0.2  # You can tune this weight later

        for x_b, c_b in tqdm(train_loader, desc="Batches", leave=False):
            #  Forward pass through encoder-decoder 
            mu, logv = enc(x_b, c_b)
            z = reparam(mu, logv)
            x_hat = dec(z, c_b)

            #  Labels for adversarial loss 
            real_labels = torch.ones(x_b.size(0), 1, device=x_b.device)
            fake_labels = torch.zeros(x_b.size(0), 1, device=x_b.device)

            #  Train Discriminator 
            dis_real = dis(x_b, c_b)
            dis_fake = dis(x_hat.detach(), c_b)
            loss_dis = F.binary_cross_entropy(dis_real, real_labels) + \
                       F.binary_cross_entropy(dis_fake, fake_labels)
            opt_dis.zero_grad()
            loss_dis.backward()
            opt_dis.step()

            #  Generator adversarial loss 
            dis_fake_for_gen = dis(x_hat, c_b)
            adv_loss = F.binary_cross_entropy(dis_fake_for_gen, real_labels)

            #  VAE reconstruction + KL loss 
            recon = weighted_mse_loss(x_hat, x_b, weight_pos=13.0, weight_zero=1.0)
            kl = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()

            #  Total VAE-GAN loss 
            loss = recon_alpha * recon + kl_beta * kl_weight * kl + adv_alpha * adv_loss

            #  Optimize encoder and decoder 
            opt_e.zero_grad(); opt_d.zero_grad()
            loss.backward()
            opt_e.step(); opt_d.step()

            sched_e.step(); sched_d.step()
            epoch_loss += loss.item() * x_b.size(0)

        avg_train = epoch_loss / len(train_loader.dataset)
        tqdm.write(f"Epoch {epoch} train loss: {avg_train:.4f}")

        #  Validation 
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

    train_true_full, train_pred_full = collect(train_loader)
    val_true_full, val_pred_full = collect(val_loader)
    test_true_full, test_pred_full = collect(test_loader)

    train_pred_full[train_true_full == 0.0] = 0.0
    val_pred_full[val_true_full == 0.0] = 0.0
    test_pred_full[test_true_full == 0.0] = 0.0

    all_rmse, ignored, kept = [], [], []
    for idx, gene in enumerate(gene_names_all):
        rmse = np.sqrt(mean_squared_error(train_true_full[:, idx], train_pred_full[:, idx]))
        entry = {"Gene": gene, "Train_RMSE": rmse}
        all_rmse.append(entry)
        (ignored if rmse > 1.0 else kept).append(entry)

    pd.DataFrame(all_rmse).to_csv(stats_dir/"train_gene_rmse_scores.csv", index=False)
    pd.DataFrame(ignored).to_csv(stats_dir/"ignored_high_rmse_genes.csv", index=False)

    gene_names[:] = [e["Gene"] for e in kept]
    kept_idx = [i for i, g in enumerate(gene_names_all) if g in gene_names]

    val_true = val_true_full[:, kept_idx]
    val_pred = val_pred_full[:, kept_idx]
    test_true = test_true_full[:, kept_idx]
    test_pred = test_pred_full[:, kept_idx]
    train_true = train_true_full[:, kept_idx]
    train_pred = train_pred_full[:, kept_idx]

    #  RMSE and R² 
    def evaluate_split(name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2s = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
        r2_df = pd.DataFrame({"Gene": gene_names, "R2": r2s})
        r2_df.to_csv(stats_dir / f"{name}_gene_r2_scores.csv", index=False)
        print(f"{name} RMSE: {rmse:.4f}")
        print(f"{name} R² Avg: {r2_df['R2'].mean():.4f}")

    evaluate_split("Training", train_true, train_pred)
    evaluate_split("Validation", val_true, val_pred)
    evaluate_split("Test", test_true, test_pred)

    for i, gene in enumerate(gene_names):
        plot_histogram(val_true[:, i], val_pred[:, i], gene, plot_dir)
    print(f"Saved histograms to: {plot_dir}")

    synth = []
    with torch.no_grad():
        for c_vec in torch.tensor(np.vstack([Ctr, Cv, Cte])):
            z = torch.randn(1, latent_dim)
            pred = dec(z, c_vec.unsqueeze(0)).squeeze(0).cpu().numpy()
            synth.append(pred)

    synth = np.clip(np.vstack(synth), 0, None)
    synth_inv = scaler.inverse_transform(synth)[:, kept_idx]
    zero_pattern_kept = zero_pattern[:, kept_idx]
    synth_inv[zero_pattern_kept] = 0.0
    synth_inv[np.abs(synth_inv) < 1e-5] = 0.0

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
