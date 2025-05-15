import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def get_column_lists(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    all_cols = df.columns.tolist()
    gene_cols = df.columns[11:]  # Assuming gene data starts at column L (12th column)
    return {
        "condition_cols": all_cols[:11],
        "gene_cols": gene_cols.tolist()
    }

# Uncomment to test manually:
# print(get_column_lists("C:/Users/huong/Desktop/Chou_Lab/VAE_GAN/NP-PC Database(Part).xlsx"))

data_dir = "C:/Users/huong/Desktop/Chou_Lab/VAE_GAN"
excel_path = os.path.join(data_dir, "NP-PC Database(Part).xlsx")
latent_dim = 16
batch_size = 128

def excel_colname_to_index(name):
    index = 0
    for c in name:
        index = index * 26 + (ord(c.upper()) - ord('A') + 1)
    return index - 1

def run_vae_gan(condition_columns, gene_columns):
    df_input = pd.read_excel(excel_path)
    df_input.columns = df_input.columns.str.strip()

    gene_col_indices = [excel_colname_to_index(col) for col in gene_columns]
    gene_col_names = df_input.columns[gene_col_indices]
    gene_expr_data = df_input.iloc[:, gene_col_indices].values.astype(np.float32)
    expr_dim = gene_expr_data.shape[1]

    df_selected = df_input[condition_columns + list(gene_col_names)]
    df_selected.to_csv(os.path.join(data_dir, "input.csv"), index=False)

    encoders = {}
    for col in condition_columns:
        unique_vals = df_input[col].unique().tolist()
        encoders[col] = {
            val: [1 if i == j else 0 for j in range(len(unique_vals))]
            for i, val in enumerate(unique_vals)
        }

    condition_data = []
    for _, row in df_input.iterrows():
        cond_vec = []
        for col in condition_columns:
            val = row[col]
            cond_vec += encoders[col][val]
        condition_data.append(cond_vec)

    condition_data = np.array(condition_data).astype(np.float32)

    class GeneExpressionDataset(Dataset):
        def __init__(self, x, c):
            self.x = torch.tensor(x)
            self.c = torch.tensor(c)
        def __len__(self):
            return len(self.x)
        def __getitem__(self, idx):
            return self.x[idx], self.c[idx]

    dataset = GeneExpressionDataset(gene_expr_data, condition_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cond_dim = condition_data.shape[1]

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(expr_dim + cond_dim, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc_mu = nn.Linear(128, latent_dim)
            self.fc_logvar = nn.Linear(128, latent_dim)
            self.relu = nn.ReLU()
            self.batch_norm = nn.BatchNorm1d(256)
        def forward(self, x, c):
            h = self.relu(self.fc1(torch.cat([x, c], dim=1)))
            h = self.batch_norm(h)
            h = self.relu(self.fc2(h))
            return self.fc_mu(h), self.fc_logvar(h)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(latent_dim + cond_dim, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, expr_dim)
            self.relu = nn.ReLU()
        def forward(self, z, c):
            h = self.relu(self.fc1(torch.cat([z, c], dim=1)))
            h = self.relu(self.fc2(h))
            h = self.relu(self.fc3(h))
            return self.fc4(h)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(expr_dim + cond_dim, 256)
            self.fc2 = nn.Linear(256, 1)
        def forward(self, x, c):
            h = F.leaky_relu(self.fc1(torch.cat([x, c], dim=1)), 0.2)
            return torch.sigmoid(self.fc2(h))

    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()
    opt_enc = optim.AdamW(encoder.parameters(), lr=0.001, weight_decay=0.01)
    opt_dec = optim.AdamW(decoder.parameters(), lr=0.001, weight_decay=0.01)
    opt_disc = optim.AdamW(discriminator.parameters(), lr=0.001, weight_decay=0.01)

    for epoch in range(20):  # Use fewer epochs for UI speed
        for x, c in dataloader:
            mu, logvar = encoder(x, c)
            z = reparameterize(mu, logvar)
            x_hat = decoder(z, c)

            real_pred = discriminator(x, c)
            fake_pred = discriminator(x_hat.detach(), c)
            disc_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred)) + \
                        F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

            fake_pred = discriminator(x_hat, c)
            gen_loss = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))
            recon_loss = F.mse_loss(x_hat, x)
            kl_loss = -0.1 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = 10 * recon_loss + kl_loss + gen_loss

            opt_enc.zero_grad()
            opt_dec.zero_grad()
            total_loss.backward()
            opt_enc.step()
            opt_dec.step()

    encoder.eval()
    decoder.eval()
    generated_data = []
    with torch.no_grad():
        for c in torch.tensor(condition_data):
            z = torch.randn(1, latent_dim)
            x_pred = decoder(z, c.unsqueeze(0)).squeeze(0).numpy()
            generated_data.append(x_pred)

    df_pred_values = pd.DataFrame(generated_data, columns=gene_col_names)
    df_conditions = df_input[condition_columns].iloc[:len(df_pred_values)].reset_index(drop=True)
    df_output = pd.concat([df_conditions, df_pred_values], axis=1)
    return df_output
