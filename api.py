from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
import os

from VAE_GAN_model import encoder, decoder, condition_data, gene_col_names, latent_dim

app = FastAPI()

class InputIndex(BaseModel):
    index: int

@app.get("/")
def read_root():
    return {"status": "API is working"}

@app.post("/predict/")
def predict(input_data: InputIndex):
    encoder.eval()
    decoder.eval()
    idx = input_data.index
    c = torch.tensor(condition_data[idx]).float().unsqueeze(0)
    z = torch.randn(1, latent_dim)
    with torch.no_grad():
        x_pred = decoder(z, c).squeeze(0).numpy()
    result = dict(zip(gene_col_names, x_pred.tolist()))
    return result
