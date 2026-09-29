from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
import json
import torch

from config import *
from src.data import get_valid_indices, WeatherDataset
from src.model import WeatherCNN
from src.train import training_model
from torch.utils.data import DataLoader

# open the normalised data

print(f"Loading the data.")

ds_normed_zarr = xr.open_zarr(DATA_DIR / "era5_ml_ready.zarr")

all_data = torch.tensor(
    ds_normed_zarr["data"].values,
    dtype=torch.float32
)

print(all_data.shape)

# prepare the training and validation datasets 
print(f"Preparing the training and validation datasets.")

times = ds_normed_zarr.forecast_reference_time.values

train_indices = get_valid_indices(times, TRAIN_START, TRAIN_END)
val_indices = get_valid_indices(times, VAL_START, VAL_END)

train_indices = train_indices[train_indices + 4 < len(all_data)]
val_indices   = val_indices[val_indices + 4 < len(all_data)]

train_dataset = WeatherDataset(all_data,train_indices)
val_dataset = WeatherDataset(all_data, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# train the model
print(f"Training the model.")

model = WeatherCNN()

training_model(model, train_loader, val_loader, epochs=50)

print(f"Model saved to models directory.")
