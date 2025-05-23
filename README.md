# VAE-GAN Gene Expression Model

This project implements a Conditional Variational Autoencoder - Generative Adversarial Network (VAE-GAN) for generating gene expression profiles based on selected condition variables from a provided Excel dataset. It supports research tasks such as synthetic gene expression generation, model training, prediction, and visualization.

## Project Structure

```
VAE-GAN-Model/                  # Local Python virtual environment (DO NOT upload to GitHub)
├── components.py              # Main model training and prediction script
├── main.py       
├── NP-PC Database(Part).xlsx  # Input Excel file containing experimental data
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files/folders to be excluded from Git
└── README.md                  # This file
```

## Cloning the repo & Running on a different device

If someone wants to use your model on a different laptop:

```bash
git clone https://github.com/yourusername/VAE-GAN-Model.git
cd VAE-GAN-Model
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python copy_model.py
```


### 2. Set Up a Virtual Environment

Create and activate a virtual environment in the project folder:

```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# OR
source .venv/bin/activate       # macOS/Linux
```

### 3. Install Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

## How to Run the Model

Make sure the file `NP-PC Database(Part).xlsx` is present in the root folder.

Then run the main model script:

```bash
python copy_model.py
```

During execution, the script will prompt you to enter:

- Condition column names (e.g., `Mod_Charge`, `NP_Type`)
- Gene expression columns using Excel-style letters (e.g., `L`, `M`, `N`)

After training, it will generate:

- `input.csv`: Cleaned subset of the original input
- `data_predictions.csv`: Generated gene expression predictions
- A plot comparing predicted vs. original gene expression values

## Features

- Conditional data generation with VAE-GAN
- Gene expression data normalization via `MinMaxScaler`
- Loss function balancing (reconstruction, KL-divergence, adversarial loss)
- Training loss curve visualization
- Fully functional prediction interface from conditions

## Author & License

Created by Huong Le — for academic and research use.

This project is licensed under the MIT License. See `LICENSE` file for details.
