import earthkit.data as ekd
from pathlib import Path
import xarray as xr
import numpy as np



# ============================================================
# Download data from CDS
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


# ============================================================
# Combine files to xarray
# ============================================================
    
def combine_nc_to_xarray(path, file_names):
    files = sorted(
        Path(path).glob(file_names+"*.nc")
        )

    ds = xr.open_mfdataset( 
        files,
        combine="by_coords"
        )
    return ds

# ============================================================
# Get indices to split into training, validation and test samples 
# ============================================================

def get_valid_indices(times, start, end):

    indices = []

    start = np.datetime64(start)
    end = np.datetime64(end)

    for i in range(len(times) - 4):

        # All five states must exist
        sample_times = times[i:i+5]

        if (
            sample_times[0] >= start
            and sample_times[-1] < end
        ):
            indices.append(i)

    return np.array(indices)

# ============================================================
# Get the weather dataset for training etc based on the indices 
# ============================================================

class WeatherDataset():

    def __init__(
        self,
        data,
        indices,
        target_channel=11,
    ):
        """
        data:
            Tensor of shape
            (time, channels, lat, lon)

        indices:
            Valid starting indices for this dataset.

        target_channel:
            11 = 2m temperature (2t)
        """

        self.data = data
        self.indices = indices
        self.target_channel = target_channel

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):

        idx = self.indices[i]

        # --------------------------------------------
        # Input at t = 0
        # --------------------------------------------

        x = self.data[idx]

        # --------------------------------------------
        # Future 2m temperature
        #
        # t+1 = +6h
        # t+2 = +12h
        # t+3 = +18h
        # t+4 = +24h
        # --------------------------------------------

        y = self.data[
            idx + 1 : idx + 5,
            self.target_channel,
            :, :
        ]

        return x, y