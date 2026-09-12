import torch

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