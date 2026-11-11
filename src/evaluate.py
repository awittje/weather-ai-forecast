import torch
import numpy as np

def rmse(pred, truth):
    return torch.sqrt(
        torch.mean((pred - truth) ** 2)
    )

def mae(pred, truth):
    return torch.mean(torch.abs(pred - truth))

def bias(pred, truth):
    return torch.mean(pred - truth)

def rmse_by_horizon(pred, truth):
    return torch.sqrt(
        torch.mean((pred - truth) ** 2,dim=(0, 2, 3))
    )

def mae_by_horizon(pred, truth):

    return torch.mean(
        torch.abs(pred - truth),dim=(0, 2, 3)
    )

def bias_by_horizon(pred, truth):

    return torch.mean(
        pred - truth,dim=(0, 2, 3)
    )

def spatial_rmse(pred, truth):
    """
    RMSE at each grid point.

    Input:
        pred, truth: (N, lat, lon)

    Output:
        (lat, lon)
    """
    return torch.sqrt(
        torch.mean((pred - truth) ** 2, dim=0)
    )

def spatial_bias(pred, truth):
    """
    Mean forecast error at each grid point.

    Positive = overprediction
    Negative = underprediction
    """
    return torch.mean(
        pred - truth,
        dim=0
    )

def spatial_correlation(pred, truth):
    """
    Pearson correlation at each grid point.

    Input:
        pred, truth: (N, lat, lon)

    Output:
        (lat, lon)
    """

    pred_mean = torch.mean(pred, dim=0)
    truth_mean = torch.mean(truth, dim=0)

    pred_anom = pred - pred_mean
    truth_anom = truth - truth_mean

    numerator = torch.sum(
        pred_anom * truth_anom,
        dim=0
    )

    denominator = torch.sqrt(
        torch.sum(pred_anom ** 2, dim=0)
        * torch.sum(truth_anom ** 2, dim=0)
    )

    return numerator / denominator

def correlation(pred, truth):
    """Pearson correlation over all samples and grid points."""

    pred = pred.flatten()
    truth = truth.flatten()

    pred = pred - torch.mean(pred)
    truth = truth - torch.mean(truth)

    return torch.sum(pred * truth) / torch.sqrt(
        torch.sum(pred ** 2) *
        torch.sum(truth ** 2)
    )


def evaluate_single_forecast(
    model,
    dataset,
    sample_idx,
    temperature_std=None
):

    model.eval()

    x, y = dataset[sample_idx]

    with torch.no_grad():
        prediction = model(
            x.unsqueeze(0)
        ).squeeze(0)

    # Persistence forecast
    persistence = (
        x[11]
        .unsqueeze(0)
        .repeat(4, 1, 1)
    )

    horizons = [6, 12, 18, 24]

    print(
        f"Sample: {sample_idx}"
    )

    for i, horizon in enumerate(horizons):

        cnn_rmse = rmse(
            prediction[i],
            y[i]
        ).item()

        persistence_rmse = rmse(
            persistence[i],
            y[i]
        ).item()

        if temperature_std is not None:
            cnn_rmse *= temperature_std
            persistence_rmse *= temperature_std

        print(
            f"+{horizon:2d}h | "
            f"CNN = {cnn_rmse:.3f} | "
            f"Persistence = {persistence_rmse:.3f}"
        )

    return x, y, prediction, persistence



def evaluate_forecasts(
    model,
    dataset,
    temperature_std=None
):

    model.eval()

    horizons = [6, 12, 18, 24]

    cnn_rmse = []
    persistence_rmse = []

    with torch.no_grad():

        for sample_idx in range(360):

            x, y = dataset[sample_idx]

            prediction = model(
                x.unsqueeze(0)
            ).squeeze(0)

            # Persistence forecast
            persistence = (
                x[11]
                .unsqueeze(0)
                .repeat(4, 1, 1)
            )

            cnn_rmse_step = []
            persistence_rmse_step = []

            for i in range(4):

                cnn_error = rmse(prediction[i], y[i]).item()
                persistence_error = rmse(persistence[i], y[i]).item()

                if temperature_std is not None:
                    cnn_error *= temperature_std
                    persistence_error *= temperature_std

                cnn_rmse_step.append(cnn_error)
                persistence_rmse_step.append(persistence_error)

            cnn_rmse.append(cnn_rmse_step)
            persistence_rmse.append(persistence_rmse_step)

    return {
        "horizons": horizons,
        "cnn_rmse": np.array(cnn_rmse),
        "persistence_rmse": np.array(persistence_rmse)
    }


def calculate_overall_rmse(
    model,
    dataset,
    temperature_std=None
):

    model.eval()

    horizons = [6, 12, 18, 24]

    all_predictions = []
    all_targets = []
    all_persistence = []

    with torch.no_grad():

        for sample_idx in range(360):

            x, y = dataset[sample_idx]

            prediction = model(
                x.unsqueeze(0)
            ).squeeze(0)

            # Persistence forecast
            persistence = (
                x[11]
                .unsqueeze(0)
                .repeat(4, 1, 1)
            )

            all_predictions.append(prediction)
            all_targets.append(y)
            all_persistence.append(persistence)

    # Shape: (360, 4, H, W)
    predictions = torch.stack(all_predictions)
    targets = torch.stack(all_targets)
    persistence = torch.stack(all_persistence)

    # Overall RMSE across time steps and grid cells
    cnn_rmse = torch.sqrt(
        torch.mean((predictions - targets) ** 2, dim=(0, 2, 3))
    )

    persistence_rmse = torch.sqrt(
        torch.mean((persistence - targets) ** 2, dim=(0, 2, 3))
    )

    if temperature_std is not None:
        cnn_rmse *= temperature_std
        persistence_rmse *= temperature_std

    return {
        "horizons": horizons,
        "cnn_rmse": cnn_rmse.numpy(),
        "persistence_rmse": persistence_rmse.numpy()
    }


def calculate_spatial_rmse(
    model,
    dataset,
    temperature_std=None
):

    model.eval()

    horizons = [6, 12, 18, 24]

    all_predictions = []
    all_targets = []
    all_persistence = []

    with torch.no_grad():

        for sample_idx in range(360):

            x, y = dataset[sample_idx]

            prediction = model(
                x.unsqueeze(0)
            ).squeeze(0)

            persistence = (
                x[11]
                .unsqueeze(0)
                .repeat(4, 1, 1)
            )

            all_predictions.append(prediction)
            all_targets.append(y)
            all_persistence.append(persistence)

    predictions = torch.stack(all_predictions)
    targets = torch.stack(all_targets)
    persistence = torch.stack(all_persistence)

    # Spatial RMSE: average over time, retain spatial dimensions
    cnn_spatial_rmse = torch.sqrt(
        torch.mean(
            (predictions - targets) ** 2,
            dim=0
        )
    )

    persistence_spatial_rmse = torch.sqrt(
        torch.mean(
            (persistence - targets) ** 2,
            dim=0
        )
    )

    # Difference between CNN and persistence
    rmse_difference = (
        cnn_spatial_rmse - persistence_spatial_rmse
    )

    if temperature_std is not None:
        cnn_spatial_rmse *= temperature_std
        persistence_spatial_rmse *= temperature_std
        rmse_difference *= temperature_std

    return {
        "horizons": horizons,
        "cnn_rmse": cnn_spatial_rmse.numpy(),
        "persistence_rmse": persistence_spatial_rmse.numpy(),
        "rmse_difference": rmse_difference.numpy()
    }