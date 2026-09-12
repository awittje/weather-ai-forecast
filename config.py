from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"


# ============================================================
# ERA5 DATA
# ============================================================

PRESSURE_LEVEL_VARS = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
]

PRESSURE_LEVELS = [500, 850]

SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

# North, West, South, East
# AREA = [55, 5, 47, 16] # Germany
AREA = [60, -10, 40, 20] # part of Europe


GRID = (0.5, 0.5)

# Time period
YEARS = ["2020", "2021"]

MONTHS = [
    "01", "02", "03", "04", "05", "06",
    "07", "08", "09", "10", "11", "12"
]

DAYS = [f"{d:02d}" for d in range(1, 32)]

# Every 6 hours
TIMES = [
    "00:00",
    "06:00",
    "12:00",
    "18:00",
]


# ============================================================
# PROCESSED DATA
# ============================================================

NORMALISED_ZARR = (
    PROCESSED_DATA_DIR / "era5_ml_ready.zarr"
)


# ============================================================
# VARIABLES USED BY THE MODEL
# ============================================================

INPUT_VARIABLES = [
    "t_500",
    "t_850",
    "u_500",
    "u_850",
    "v_500",
    "v_850",
    "z_500",
    "z_850",
    "10u",
    "10v",
    "2d",
    "2t",
    "msl",
]

TARGET_VARIABLE = "2t"

# +6, +12, +18, +24 hours
FORECAST_HORIZONS = [1, 2, 3, 4]


# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

TRAIN_START = "2020-01-01"
TRAIN_END = "2021-06-30"

VAL_START = "2021-07-01"
VAL_END = "2021-09-30"

TEST_START = "2021-10-01"
TEST_END = "2021-12-31"


# ============================================================
# MODEL
# ============================================================

IN_CHANNELS = len(INPUT_VARIABLES)

OUT_CHANNELS = len(FORECAST_HORIZONS)

HIDDEN_CHANNELS = 64


# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 32

#LEARNING_RATE = 1e-4

EPOCHS = 50

#WEIGHT_DECAY = 1e-5

#RANDOM_SEED = 42
