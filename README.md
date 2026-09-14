# weather-ai-forecast

# AI Weather Forecasting with ERA5

A small research project exploring **machine-learning-based regional weather forecasting using ERA5 reanalysis data and PyTorch**.

The project develops a convolutional neural network (CNN) that takes the atmospheric state at an initial time and predicts the near-future 2 m temperature field at **+6, +12, +18 and +24 hours**.

The main focus is not only on producing forecasts, but also on **systematic forecast verification** against the ERA5 reference state and comparison with a persistence baseline.

> **Project status:** Work in progress. The current version establishes the data-processing, CNN forecasting and initial verification pipeline. Further improvements to the forecasting model, uncertainty estimation and automated testing are planned.

---

## Project motivation

Modern numerical weather prediction systems provide highly accurate forecasts but are computationally expensive. Recent advances in machine learning have opened up the possibility of using data-driven models as components of weather forecasting systems.

This project explores a simplified version of this problem:

> **Can a convolutional neural network learn the evolution of a regional atmospheric state from ERA5 data and produce useful short-term temperature forecasts?**

The project is also intended as a practical exploration of the computational and methodological requirements of AI-based weather prediction, including:
* meteorological data preparation
* physical and statistical modelling
* machine learning
* forecast verification
* reproducible scientific computing
* automated workflows

---

## Current approach

The model operates on a regional domain over parts of Europe. As input, 13 atmospheric variables are used:

### Pressure-level variables

* Temperature at 500 hPa
* Temperature at 850 hPa
* U wind component at 500 hPa
* U wind component at 850 hPa
* V wind component at 500 hPa
* V wind component at 850 hPa
* Geopotential at 500 hPa
* Geopotential at 850 hPa

### Single-level variables

* 2 m temperature
* 2 m dew point temperature
* Mean sea-level pressure
* 10 m U wind component
* 10 m V wind component

The model receives the state at time `t = 0` and predicts:

```text
t + 6 h
t + 12 h
t + 18 h
t + 24 h
```

for the 2 m temperature field.

---

## Data

The project uses **ERA5 reanalysis data** from the Copernicus Climate Change Service.

The current dataset covers:

* **Period:** 2020–2021
* **Temporal resolution:** 6 hours
* **Spatial resolution:** 0.5° × 0.5°
* **Domain:** approximately 40–60°N, -10–20°E covering parts of Europe

The resulting dataset contains:

```text
2924 time steps
13 channels/variables
41 × 61 spatial grid
```

The processed dataset is stored as an **xarray/Zarr dataset** to allow efficient access without loading the complete dataset into memory.

---

## Machine-learning model

The forecasting model is a convolutional neural network implemented in **PyTorch**.

The basic concept is:

```text
                 ERA5 atmospheric state
                         │
                         ▼
              ┌─────────────────────┐
              │   Input channels    │
              │       13 × H × W    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Convolutional     │
              │      layers         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Learned spatial and │
              │ atmospheric patterns│
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Forecast output     │
              │   4 × H × W         │
              └─────────────────────┘
                         │
                         ▼
                  2 m temperature
                +6/+12/+18/+24 h
```

---

## Training and evaluation

The data are split chronologically to avoid mixing future information into the training set:

| Dataset    | Period                  |
| ---------- | ----------------------- |
| Training   | 2020-01-01 – 2021-06-30 |
| Validation | 2021-07-01 – 2021-09-30 |
| Test       | 2021-10-01 – 2021-12-31 |

The training data are shuffled **after** the chronological split.

The validation dataset is used for model development and selection, while the final test period is kept separate for evaluating forecasting performance.

---

## Forecast baseline

The CNN is compared against a simple **persistence forecast**.

Persistence assumes that the atmospheric state remains unchanged:

```text
forecast(t + Δt) = state(t)
```

For temperature, this means that the forecast at +6, +12, +18 and +24 hours is simply the temperature field at the initial time.

Persistence provides a useful baseline because a machine-learning forecasting model should demonstrate skill beyond this simple assumption.

---

## Forecast verification

The project evaluates forecasts using several complementary metrics.

### Deterministic metrics

* RMSE — Root Mean Squared Error
* MAE — Mean Absolute Error
* Bias
* Pearson correlation

### Spatial verification

The spatial structure of forecast errors is investigated using:

* spatial RMSE
* spatial bias
* spatial correlation

This allows the project to identify geographical regions where the model performs particularly well or poorly.

### Lead-time dependence

Forecast skill is evaluated separately at:

```text
+6 h
+12 h
+18 h
+24 h
```

This makes it possible to investigate how quickly forecast errors grow with increasing lead time.

### Case studies

Individual weather situations are visualized by comparing:

```text
ERA5 reference
CNN forecast
Persistence forecast
Forecast error
```

for different forecast horizons.

---

## Example output

Example temperature forecast maps can be generated for individual test cases.

The visualization compares the predicted and reference 2 m temperature fields and shows the forecast error in °C.

Further evaluation will include aggregate statistics over the complete test period and spatial error maps.

---

## Project structure

The repository is organized as follows:

```text
weather-ai-forecasts/
│
├── README.md
├── config.py
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── forecast.py
│   ├── evaluate.py
│   └── plotting.py
│
├── results/
│   ├── figures/
│   └── metrics/
│
└── notebooks/
    └── xxx.ipynb
```

### `config.py`

Central configuration of:

* ERA5 variables
* spatial domain
* temporal range
* train/validation/test periods
* model parameters
* training parameters
* file paths

### `src/data.py`

Data loading and preparation, including the PyTorch `Dataset`.

### `src/model.py`

Definition of the PyTorch CNN.

### `src/train.py`

Training and validation procedures.

### `src/forecast.py`

Generation of CNN and persistence forecasts.

### `src/evaluate.py`

Forecast verification metrics.

### `src/plotting.py`

Visualization of temperature fields, forecast errors and verification results.

### `notebooks/`

Exploratory analysis and demonstrations of the forecasting workflow.

### `results/`

Generated figures and evaluation metrics.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/awittje/weather-ai-forecasts.git
cd weather-ai-forecasts
```

Create a Python environment and install the required packages:

```bash
conda create -n weather python=3.12
conda activate weather
pip install -r requirements.txt
```

The project uses, among others:

* PyTorch
* xarray
* NumPy
* pandas
* matplotlib
* Cartopy
* Zarr
* earthkit

---

## Running the project

The workflow is currently:

```text
ERA5 data
    │
    ▼
Data preparation
    │
    ▼
Normalisation
    │
    ▼
PyTorch Dataset
    │
    ▼
CNN training
    │
    ▼
Forecast generation
    │
    ├──────────────┐
    ▼              ▼
CNN forecast   Persistence
    │              │
    └──────┬───────┘
           ▼
     Forecast verification
           │
           ▼
       Visualisation
```

---

## Future work

Planned extensions include:

### Improved forecasting model

* residual convolutional blocks
* improved loss functions
* multi-step forecasting experiments
* comparison of different network architectures
* investigation of model resolution and domain size

### Forecast verification

* full-test-period verification
* lead-time-dependent RMSE/MAE
* spatial error statistics
* weather-regime analysis
* extreme-weather case studies
* comparison against additional baselines

### Probabilistic forecasting

An important future extension is the estimation of forecast uncertainty, for example using ensemble or Monte-Carlo-dropout approaches.

This would allow investigation of:

* ensemble spread
* forecast uncertainty
* spread-skill relationships
* prediction interval coverage
* reliability

### Reproducibility and automation

The project will also be extended with:

* automated tests
* GitHub Actions / continuous integration
* reproducible training workflows
* automated evaluation
* model checkpointing
* experiment configuration and tracking

---

## Scientific context

The project is inspired by current developments in AI-based weather forecasting and the increasing use of machine-learning methods alongside traditional numerical weather prediction.

Rather than attempting to reproduce a state-of-the-art operational forecasting system, this project focuses on understanding the individual components of an AI weather forecasting workflow:

**data → model → forecast → verification → uncertainty**

The project therefore serves as a compact experimental framework for investigating how machine-learning models can learn spatial and temporal structures in atmospheric data.

---

## Author

**Anna Wittje**

Physicist / Scientific Computing / Machine Learning

This project is part of a broader transition from astrophysical data analysis and simulation-based modelling toward applications in **Earth observation, numerical weather prediction and AI-based Earth-system modelling**.

