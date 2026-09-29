from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
import json
import torch

from config import *
from torch.utils.data import Dataset

ds_merged = xr.open_dataset(DATA_DIR / 'merged.nc')

# normalise the data with the statitics from the training period

# Compute normalisation statistics (mean and std) for each variable
# In practice, you'd compute these over the full training period (e.g., 1979-2018)
# and store them alongside the dataset

ds_subset = ds_merged.sel(forecast_reference_time=slice(TRAIN_START, TRAIN_END))

stats = {}
for var in ds_subset.data_vars:
    values = ds_subset[var].values
    stats[var] = {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
    }
    print(f"{var:>12s}: mean={stats[var]['mean']:12.2f}, std={stats[var]['std']:10.2f}")

# Apply z-score normalisation
# Normalise each variable using its mean and std
# Store the result in ds_normed

ds_normed = ds_merged.copy() 
for var in ds_normed.data_vars:
    ds_normed[var] = (ds_normed[var] - stats[var]["mean"]) / stats[var]["std"]

# To verify, we plot the histograms of the normalised values 
    
print(f"Plot the histograms of the normalised statistics.")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

for idx, var in enumerate(ds_normed.data_vars):
    row, col = divmod(idx, 4)
    if row >= 2 or col >= 4:
        break
    ax = axes[row, col]
    vals = ds_normed[var].values.flatten()
    vals = vals[~np.isnan(vals)]
    ax.hist(vals, bins=100, color="tab:blue")
    ax.set_xlabel(var)
    ax.set_ylabel("Number of observations")
    ax.set_title(f"mean={vals.mean():.2f}, std={vals.std():.2f}")

for idx in range(len(list(ds_normed.data_vars)), axes.size):
    row, col = divmod(idx, 4)
    axes[row, col].set_visible(False)

PLOT_PATH = DATA_DIR / "statistics_after_normalisation.png"
plt.suptitle("Histograms after the Z-score normalisation")
plt.tight_layout()
plt.savefig(PLOT_PATH)

print(f"Plot saved to: {PLOT_PATH}")

# save the normalised xarray as a zarr file for the training of the model 

# Stack all variables into a single DataArray with shape (time, channel, lat, lon)
# Use xr.concat or np.stack to create a (time, channel, lat, lon) array

variable_names = list(ds_normed.data_vars)
print(f"Variables ({len(variable_names)}): {variable_names}")

stacked = np.stack([ds_normed[var].values for var in variable_names], axis=1) 

# stacked should have shape (n_times, n_channels, n_lat, n_lon)
print(f"\nStacked shape: {stacked.shape}")
print(f"  time steps:  {stacked.shape[0]}")
print(f"  channels:    {stacked.shape[1]}")
print(f"  lat x lon:   {stacked.shape[2]} x {stacked.shape[3]}")

# Save to Zarr with ML-friendly chunking
# TODO: Create an xarray Dataset from the stacked array and save to Zarr
# Use chunks of (1, n_channels, n_lat, n_lon) -- one full field per time step

# Also save the normalisation statistics as attributes so we can denormalise later

ds_out = xr.Dataset(
    {
        "data": xr.DataArray(
            stacked,
            dims=["forecast_reference_time", "channel", "latitude", "longitude"],
            coords={
                "forecast_reference_time": ds_normed.forecast_reference_time.values,
                "channel": variable_names,
                "latitude": ds_normed.latitude.values,
                "longitude": ds_normed.longitude.values,
            },
        )
    },
    attrs={
        "normalisation_stats": json.dumps(stats),
        "description": "ML-ready ERA5 subset, z-score normalised",
    },
)

# Chunk: 1 time step, all channels, all spatial
ds_out = ds_out.chunk(
    {
        "forecast_reference_time": 1,
        "channel": len(variable_names),
        "latitude": len(ds_normed.latitude),
        "longitude": len(ds_normed.longitude),
    }
)

ZARR_PATH = DATA_DIR / "era5_ml_ready.zarr"

ds_out.to_zarr(str(ZARR_PATH), mode="w")
print(f"Saved to {ZARR_PATH}")

