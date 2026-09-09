#import time
#from pathlib import Path

import earthkit.data as ekd
#import earthkit.plots as ekp
#import matplotlib.pyplot as plt
#import numpy as np
#import xarray as xr
#import zarr


# ============================================================
# Download functions
# ============================================================

def download_pressure_levels(year, month, days, area, times, grid, pressure_level_vars, pressure_levels, parent_dir):
    """Download one month of ERA5 pressure-level data."""

    output_file = parent_dir / f"era5_pl_{year}_{month}.nc"

    # --------------------------------------------------------
    # Skip if file already exists
    # --------------------------------------------------------
    if output_file.exists():
        print(f"[SKIP] {output_file} already exists.")
        return

    print(f"[DOWNLOAD] Pressure levels: {year}-{month}")

    data = ekd.from_source(
        "cds",
        "reanalysis-era5-pressure-levels",

        variable=pressure_level_vars,
        pressure_level=pressure_levels,

        product_type="reanalysis",

        year=year,
        month=month,
        day=days,

        area=area,
        time=times,

        grid=grid,
    )

    data.save(str(output_file))

    print(f"[DONE] {output_file}")


def download_single_levels(year, month, days, area, times, grid, single_level_vars, parent_dir):
    """Download one month of ERA5 single-level data."""

    output_file = parent_dir / f"era5_sl_{year}_{month}.nc"

    # --------------------------------------------------------
    # Skip if file already exists
    # --------------------------------------------------------
    if output_file.exists():
        print(f"[SKIP] {output_file} already exists.")
        return

    print(f"[DOWNLOAD] Single levels: {year}-{month}")

    data = ekd.from_source(
        "cds",
        "reanalysis-era5-single-levels",

        variable=single_level_vars,

        product_type="reanalysis",

        year=year,
        month=month,
        day=days,

        area=area,
        time=times,

        grid=grid,
    )

    data.save(str(output_file))

    print(f"[DONE] {output_file}")
