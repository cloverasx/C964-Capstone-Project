"""
Training utilities for the Vehicle Identification System.

This module provides utility classes and functions for training, including:
- Custom loss functions
- Metric tracking and visualization
- Model checkpointing
- Training criteria and analysis
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization"""

    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class MetricTracker:
    """Tracks and logs training metrics"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_make_acc": [],
            "val_make_acc": [],
            "train_model_acc": [],
            "val_model_acc": [],
            "train_year_acc": [],
            "val_year_acc": [],
            "learning_rates": [],
        }

    def update(self, metrics, phase="train"):
        """Update metrics for current epoch"""
        for k, v in metrics.items():
            self.history[f"{phase}_{k}"].append(v)

    def plot_metrics(self):
        """Create and save training plots"""
        # Loss plot
        plt.figure(figsize=(12, 4))
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.output_dir / "loss_plot.png")
        plt.close()

        # Accuracy plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Make accuracy
        axes[0].plot(self.history["train_make_acc"], label="Train")
        axes[0].plot(self.history["val_make_acc"], label="Val")
        axes[0].set_title("Make Accuracy")
        axes[0].legend()

        # Model accuracy
        axes[1].plot(self.history["train_model_acc"], label="Train")
        axes[1].plot(self.history["val_model_acc"], label="Val")
        axes[1].set_title("Model Accuracy")
        axes[1].legend()

        # Year accuracy
        axes[2].plot(self.history["train_year_acc"], label="Train")
        axes[2].plot(self.history["val_year_acc"], label="Val")
        axes[2].set_title("Year Accuracy")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_plots.png")
        plt.close()


def compute_metrics(outputs, labels):
    """Compute accuracy metrics for all tasks"""
    metrics = {}

    for task in ["make", "model", "year"]:
        preds = outputs[task].argmax(dim=1)
        correct = (preds == labels[task]).float().sum()
        metrics[f"{task}_acc"] = (correct / len(preds)).item()

    return metrics


class ModelCheckpoint:
    """Handles model checkpointing"""

    def __init__(self, output_dir, mode="min", verbose=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.verbose = verbose
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def __call__(self, value, model, epoch, optimizer=None, scheduler=None):
        improved = False

        if self.mode == "min":
            improved = value < self.best_value
        else:
            improved = value > self.best_value

        if improved:
            self.best_value = value
            self.save_checkpoint(model, epoch, optimizer, scheduler)
            if self.verbose:
                print(f"Checkpoint saved! Best value: {self.best_value:.4f}")

        return improved

    def save_checkpoint(self, model, epoch, optimizer=None, scheduler=None):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_value": self.best_value,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Save with datetime in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        dated_filename = f"best_model_{timestamp}.pt"
        torch.save(checkpoint, self.output_dir / dated_filename)

        # Also save as best_model.pt for compatibility
        torch.save(checkpoint, self.output_dir / "best_model.pt")


class TrainingCriteria:
    def __init__(self, config):
        self.min_epochs = config.get("min_epochs", 10)
        self.max_epochs = config.get("max_epochs", 100)
        self.early_stop_patience = config.get("patience", 10)
        self.min_validation_acc = config.get("min_val_acc", 0.80)
        self.convergence_threshold = config.get("conv_threshold", 0.001)

    def should_stop(self, epoch, val_metrics, patience_counter, prev_val_metrics=None):
        """Determine if training should stop based on multiple criteria"""
        # Must run for minimum epochs
        if epoch < self.min_epochs:
            return False

        # Stop if reached maximum epochs
        if epoch >= self.max_epochs:
            return True

        # Early stopping check
        if patience_counter >= self.early_stop_patience:
            print(
                "Early stopping: No improvement for {} epochs".format(
                    self.early_stop_patience
                )
            )
            return True

        # Check if reached minimum validation accuracy
        if val_metrics["make_acc"] >= self.min_validation_acc:
            print("Reached minimum validation accuracy target")
            return True

        # Check convergence if we have previous metrics
        if prev_val_metrics is not None:
            improvement = abs(val_metrics["loss"] - prev_val_metrics["loss"])
            if improvement < self.convergence_threshold:
                print("Model converged: improvement below threshold")
                return True

        return False


class TrainingAnalyzer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.val_losses = []
        self.val_accuracies = []

    def update(self, val_loss, val_acc):
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

    def is_converging(self):
        """Check if training is converging"""
        if len(self.val_losses) < self.window_size:
            return False

        recent_losses = self.val_losses[-self.window_size :]
        loss_trend = np.polyfit(range(self.window_size), recent_losses, 1)[0]

        # Check if loss is plateauing
        return abs(loss_trend) < 0.001

    def is_overfitting(self):
        """Check for signs of overfitting"""
        if len(self.val_losses) < self.window_size:
            return False

        recent_losses = self.val_losses[-self.window_size :]
        recent_accs = self.val_accuracies[-self.window_size :]

        # Check if validation loss is increasing while accuracy stagnates
        loss_trend = np.polyfit(range(self.window_size), recent_losses, 1)[0]
        acc_trend = np.polyfit(range(self.window_size), recent_accs, 1)[0]

        return loss_trend > 0 and acc_trend < 0.001
