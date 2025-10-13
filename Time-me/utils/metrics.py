import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(predictions, targets):
    """Calculate comprehensive evaluation metrics"""
    # Flatten arrays
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # Calculate metrics
    mse = mean_squared_error(target_flat, pred_flat)
    mae = mean_absolute_error(target_flat, pred_flat)
    rmse = np.sqrt(mse)
    mape = calculate_mape(target_flat, pred_flat)
    r2 = r2_score(target_flat, pred_flat)

    # Calculate directional accuracy
    directional_acc = calculate_directional_accuracy(predictions, targets)

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_acc
    }


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_directional_accuracy(predictions, targets):
    """Calculate directional accuracy for time series"""
    if len(predictions.shape) == 3:
        # If predictions are [batch, seq_len, features], calculate for each sequence
        pred_changes = np.diff(predictions, axis=1)
        target_changes = np.diff(targets, axis=1)

        # Count matching directions
        correct_directions = np.sum(np.sign(pred_changes) == np.sign(target_changes))
        total_directions = pred_changes.size

        return correct_directions / total_directions if total_directions > 0 else 0
    else:
        # For flat arrays
        pred_changes = np.diff(predictions.flatten())
        target_changes = np.diff(targets.flatten())

        correct_directions = np.sum(np.sign(pred_changes) == np.sign(target_changes))
        total_directions = len(pred_changes)

        return correct_directions / total_directions if total_directions > 0 else 0


def calculate_smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return np.mean(numerator[mask] / denominator[mask]) * 100


def calculate_mase(y_true, y_pred, seasonal_period=1):
    """Calculate Mean Absolute Scaled Error"""
    if len(y_true) <= seasonal_period:
        return np.nan

    # Calculate naive forecast (seasonal naive)
    if seasonal_period == 1:
        # Simple naive forecast
        naive_forecast = y_true[:-1]
        naive_target = y_true[1:]
    else:
        # Seasonal naive forecast
        naive_forecast = y_true[:-seasonal_period]
        naive_target = y_true[seasonal_period:]

    # Calculate MAE of naive forecast
    naive_mae = np.mean(np.abs(naive_target - naive_forecast))

    # Calculate MAE of predictions
    pred_mae = np.mean(np.abs(y_true - y_pred))

    # Calculate MASE
    return pred_mae / naive_mae if naive_mae != 0 else np.nan


def evaluate_model(model, data_loader, device, criterion=None):
    """Comprehensive model evaluation"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            if criterion is not None:
                loss = criterion(output, target)
                total_loss += loss.item()

            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)

    # Calculate average loss
    avg_loss = total_loss / len(data_loader) if criterion is not None else None

    return {
        'loss': avg_loss,
        'metrics': metrics,
        'predictions': all_predictions,
        'targets': all_targets
    }


def calculate_per_variable_metrics(predictions, targets):
    """Calculate metrics for each variable separately"""
    if predictions.shape[-1] != targets.shape[-1]:
        raise ValueError("Predictions and targets must have the same number of variables")

    n_variables = predictions.shape[-1]
    variable_metrics = {}

    for i in range(n_variables):
        pred_var = predictions[..., i]
        target_var = targets[..., i]

        var_metrics = calculate_metrics(pred_var, target_var)
        variable_metrics[f'variable_{i}'] = var_metrics

    return variable_metrics


def calculate_stability_metrics(predictions, targets):
    """Calculate stability and robustness metrics"""
    # Calculate prediction variance
    pred_variance = np.var(predictions, axis=0)

    # Calculate prediction bias
    pred_bias = np.mean(predictions - targets, axis=0)

    # Calculate prediction error variance
    errors = predictions - targets
    error_variance = np.var(errors, axis=0)

    # Calculate signal-to-noise ratio
    signal_power = np.var(targets, axis=0)
    noise_power = error_variance
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    return {
        'prediction_variance': pred_variance,
        'prediction_bias': pred_bias,
        'error_variance': error_variance,
        'snr_db': snr
    }


def print_metrics(metrics, prefix=""):
    """Print metrics in a readable format"""
    print(f"\n{prefix}Metrics:")
    print("-" * 50)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key:20s}: {value:.6f}")
        else:
            print(f"{key:20s}: {value}")
    print("-" * 50)