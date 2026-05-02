import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class StreamingMetricsAccumulator:
    """Accumulate regression metrics without materializing the full dataset."""

    def __init__(self):
        self.sse = 0.0
        self.sae = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.count = 0
        self.mape_sum = 0.0
        self.mape_count = 0
        self.directional_correct = 0
        self.directional_total = 0

    def update(self, predictions, targets):
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)

        errors = predictions - targets
        self.sse += np.square(errors, dtype=np.float64).sum(dtype=np.float64)
        self.sae += np.abs(errors).sum(dtype=np.float64)
        self.sum_y += targets.sum(dtype=np.float64)
        self.sum_y2 += np.square(targets, dtype=np.float64).sum(dtype=np.float64)
        self.count += targets.size

        non_zero_mask = targets != 0
        if np.any(non_zero_mask):
            pct_errors = np.abs(errors[non_zero_mask] / targets[non_zero_mask])
            self.mape_sum += pct_errors.sum(dtype=np.float64)
            self.mape_count += int(non_zero_mask.sum())

        if predictions.ndim == 3:
            pred_changes = np.diff(predictions, axis=1)
            target_changes = np.diff(targets, axis=1)
        else:
            pred_changes = np.diff(predictions.reshape(-1))
            target_changes = np.diff(targets.reshape(-1))
        self.directional_correct += int(np.sum(np.sign(pred_changes) == np.sign(target_changes)))
        self.directional_total += pred_changes.size

    def compute(self):
        if self.count == 0:
            raise ValueError("No samples were accumulated for metric computation")

        mse = self.sse / self.count
        mae = self.sae / self.count
        rmse = np.sqrt(mse)
        mape = (self.mape_sum / self.mape_count) * 100 if self.mape_count > 0 else np.nan
        sst = self.sum_y2 - (self.sum_y ** 2) / self.count
        r2 = 1.0 - (self.sse / sst) if sst > 0 else np.nan
        directional_acc = (
            self.directional_correct / self.directional_total
            if self.directional_total > 0 else 0.0
        )

        return {
            'MSE': float(mse),
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'R2': float(r2),
            'Directional_Accuracy': float(directional_acc)
        }


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
