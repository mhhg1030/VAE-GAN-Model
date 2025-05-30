import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import re
from pathlib import Path

def plot_histogram(true_vals, pred_vals, gene, plot_dir):
    safe_gene_name = re.sub(r'[\\/*?:"<>|]', "_", gene)
    plt.figure(figsize=(6, 4))
    sns.histplot(true_vals, label='True', color='blue', stat='density', bins=30, kde=True, alpha=0.5)
    sns.histplot(pred_vals, label='Predicted', color='orange', stat='density', bins=30, kde=True, alpha=0.5)
    plt.title(f"{gene} – True vs Predicted")
    plt.xlabel("Gene Expression")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(plot_dir) / f"{safe_gene_name.replace(' ', '_')}_hist.png")
    plt.close()

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
    plt.savefig(Path(save_dir) / "pca_real_vs_synthetic.png")
    plt.close()

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000)
    X_tsne = tsne.fit_transform(combined)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, alpha=0.6)
    plt.title("t-SNE: Real vs. Synthetic")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "tsne_real_vs_synthetic.png")
    plt.close()

    print("Saved PCA and t-SNE plots comparing real vs. synthetic data.")
