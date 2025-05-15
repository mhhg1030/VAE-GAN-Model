import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Setup
data_dir = "C:/Users/huong/Desktop/Chou_Lab/VAE_GAN"
excel_path = os.path.join(data_dir, "NP-PC Database(Part).xlsx")
latent_dim = 16
batch_size = 128

# === User input ===
print("\nEnter condition column names (e.g., Mod_Charge, NP_Type). Type 'done' when finished.\n")
condition_columns = []
while True:
    col = input("Condition column (or 'done'): ").strip()
    if col.lower() == "done":
        break
    condition_columns.append(col)

print("\nEnter gene column letters starting from column 'L' onward (e.g., L, M, N). Type 'done' when finished.\n")
gene_columns = []
while True:
    col = input("Gene column (or 'done'): ").strip()
    if col.lower() == "done":
        break
    gene_columns.append(col)

# === Load Excel and process ===
df_input = pd.read_excel(excel_path)
df_input.columns = df_input.columns.str.strip()

# Ensure the condition columns exist
for col in condition_columns:
    if col not in df_input.columns:
        raise ValueError(f"Column '{col}' not found in Excel file.")

# Convert Excel letter to column index
def excel_colname_to_index(name):
    index = 0
    for c in name:
        index = index * 26 + (ord(c.upper()) - ord('A') + 1)
    return index - 1

# Process gene columns
gene_col_indices = [excel_colname_to_index(col) for col in gene_columns]
gene_col_names = df_input.columns[gene_col_indices]
gene_expr_data = df_input.iloc[:, gene_col_indices].values.astype(np.float32)
expr_dim = gene_expr_data.shape[1]

# Save selected input data to CSV
df_selected = df_input[condition_columns + list(gene_col_names)]
df_selected.to_csv(os.path.join(data_dir, "input.csv"), index=False)
print("Saved selected input data to 'input.csv'")

# One-hot encode conditions
encoders = {}
for col in condition_columns:
    unique_vals = df_input[col].unique().tolist()
    encoders[col] = {
        val: [1 if i == j else 0 for j in range(len(unique_vals))]
        for i, val in enumerate(unique_vals)
    }

# Build condition vectors
condition_data = []
for _, row in df_input.iterrows():
    cond_vec = []
    for col in condition_columns:
        val = row[col]
        cond_vec += encoders[col][val]
    condition_data.append(cond_vec)

condition_data = np.array(condition_data).astype(np.float32)

# Dataset
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

# === Improved Model components ===
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
        self.fc1 = nn.Linear(latent_dim + cond_dim, 512)  # Input size: latent_dim + cond_dim
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, expr_dim)
        self.relu = nn.ReLU()

    def forward(self, z, c):
        h = self.relu(self.fc1(torch.cat([z, c], dim=1)))  # Concatenate z and c, input to fc1
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

# Initialize models with better architecture
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()
opt_enc = optim.AdamW(encoder.parameters(), lr=0.001, weight_decay=0.01)
opt_dec = optim.AdamW(decoder.parameters(), lr=0.001, weight_decay=0.01)
opt_disc = optim.AdamW(discriminator.parameters(), lr=0.001, weight_decay=0.01)

# === Training loop with improved techniques ===
for epoch in range(200):
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

    print(f"Epoch {epoch:03d}: Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f}, Gen={gen_loss.item():.4f}, Disc={disc_loss.item():.4f}")

# === Prediction ===
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
df_output.to_csv(os.path.join(data_dir, "data_predictions.csv"), index=False)
print("Saved predictions with condition columns to 'data_predictions.csv'")

# === Plot: Predicted vs Original ===
df_pred = pd.read_csv(os.path.join(data_dir, "data_predictions.csv"))
df_input_csv = pd.read_csv(os.path.join(data_dir, "input.csv"))

gene_cols = list(gene_col_names)
min_len = min(len(df_input_csv), len(df_pred))
y_true = df_input_csv[gene_cols].iloc[:min_len].values.flatten()
y_pred = df_pred[gene_cols].iloc[:min_len].values.flatten()

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.6, s=10)
# Get max range across both axes
max_val = max(y_true.max(), y_pred.max())
min_val = min(y_true.min(), y_pred.min())

# Plot the ideal diagonal
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y = x)')


# Set axis labels
plt.xlabel("Original Expression Values")
plt.ylabel("Predicted Expression Values")
plt.title("Predicted vs. Original Gene Expression")

# Set the y-axis limits and ticks
y_max = y_pred.max()  # Get the max value for y
plt.ylim(0, y_max + 2)  # Set the y-axis limit to max value + 10
plt.yticks(np.arange(0, y_max + 2, 0.5))  # Set y-ticks from 0 to max value + 10, with step of 2

plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

# expose prediction function
import torch
import numpy as np

def predict_from_index(index):
    encoder.eval()
    decoder.eval()
    c = torch.tensor(condition_data[index]).float().unsqueeze(0)
    z = torch.randn(1, 16)
    with torch.no_grad():
        x_pred = decoder(z, c).squeeze(0).numpy()
    return dict(zip(gene_col_names, x_pred.tolist()))
