# VAE-GAN Gene Expression Model
Conditional Variational Autoencoder–Generative Adversarial Network (VAE-GAN) for generating synthetic gene-expression profiles based on selected experimental conditions.

## Repository Structure
```
VAE-GAN-Model/
├── .venv/ # Local Python virtual environment (ignored)
├── pycache/ # Python cache files (ignored)
├── data/
│ └── NP-PC Database(Part).xlsx # Excel dataset with condition & gene columns
├── components.py # Dataset class, encoder/decoder, loss functions
├── main.py # Training, evaluation & prediction workflow
├── requirements.txt # Python dependencies
└── .gitignore # Files/folders excluded from Git
```
> **Note:** A `plots/` directory will be created at runtime to store training-and-evaluation visualizations.

---

## Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/VAE-GAN-Model.git
   cd VAE-GAN-Model
2. **Create & activate a virtual environment**
   ```bash
    python -m venv .venv
   ```
    **macOS/Linux**
   ```
    source .venv/bin/activate
   ```
    **Windows**
   ```
    .\.venv\Scripts\activate
   ```
4. **Install dependencies**
    pip install -r requirements.txt

## Running the model
``
    Make sure NP-PC Database(Part).xlsx is in the data/ folder, then run: 

        python main.py

``
    You’ll be prompted to enter:

        Condition columns (e.g. Mod_Charge, NP_Type)
        Gene-expression columns by Excel-style letters (e.g. L, M, N)
``
    Outputs:

        input.csv – cleaned subset of your original data
        data_predictions.csv – generated gene-expression profiles
        Plots (loss curves, predicted vs. original) in plots/


## Key Features
``
    Conditional generation via VAE-GAN
    Normalization with MinMaxScaler
    Balanced losses: reconstruction, KL divergence, adversarial
    Interactive prompts to choose conditions & genes
    Built-in visualization of training progress & fidelity

## License
``
    MIT License – see the included LICENSE for details.
    Created by Huong Le for academic & research use.
