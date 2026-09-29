from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from config import *
from src.data import download_pressure_levels, download_single_levels, combine_nc_to_xarray

### create the paths 
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

PL_DIR=RAW_DATA_DIR / "pressure_levels"
SL_DIR=RAW_DATA_DIR / "single_levels"

PL_DIR.mkdir(parents=True, exist_ok=True)
SL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Main
# ============================================================

### the variables are taken from config.py 

def main():

    for year in YEARS:

        for month in MONTHS:

            print()
            print("=" * 60)
            print(f"Processing {year}-{month}")
            print("=" * 60)

            try:
                download_pressure_levels(year, month, DAYS, AREA, TIMES, GRID, PRESSURE_LEVEL_VARS, PRESSURE_LEVELS, PL_DIR)
                download_single_levels(year, month, DAYS, AREA, TIMES, GRID,SINGLE_LEVEL_VARS, SL_DIR)

            except Exception as error:

                print()
                print(f"[ERROR] Failed for {year}-{month}")
                print(error)
                print()
                print("Continuing with the next month...")


if __name__ == "__main__":
    main()


# combining the .nc files and converting to xarrays

ds_pl = combine_nc_to_xarray(PL_DIR, "era5_pl_")
ds_sl = combine_nc_to_xarray(SL_DIR, "era5_sl_")

# Flatten pressure level variables: t@500, t@700, t@850, t@1000, u@500, ...
# Create a new dataset where each var-level pair is a separate variable
# e.g., ds_pl["t"].sel(level=500) --> ds_flat["t_500"]

flat_vars = {}
for var in ds_pl.data_vars:
    for level in ds_pl.level.values:
        name = f"{var}_{int(level)}"
        flat_vars[name] = ds_pl[var].sel(level=level).drop_vars("level")

ds_flat = xr.Dataset(flat_vars)

# Now merge with single-level variables
# Combine ds_flat and ds_sl into one dataset

ds_merged = xr.merge([ds_flat, ds_sl])
ds_merged.to_netcdf(DATA_DIR / 'merged.nc', mode='w')


print(f"Merged dataset: {len(ds_merged.data_vars)} variables")
print(list(ds_merged.data_vars))
print()
print(ds_merged)