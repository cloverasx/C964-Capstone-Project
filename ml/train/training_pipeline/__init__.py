"""
Vehicle Identification System Training Pipeline

This package contains all the components needed for training the vehicle
classification model, including:
- Dataset handling and preprocessing
- Training orchestration and optimization
- Metrics tracking and evaluation
- Model checkpointing and early stopping
"""

from .trainer import VehicleTrainer
from .dataset import VehicleDataset
from .model import VehicleClassifier
from .training_utils import (
    LabelSmoothingLoss,
    MetricTracker,
    compute_metrics,
    ModelCheckpoint,
    TrainingCriteria,
    TrainingAnalyzer,
)

__all__ = [
    "VehicleTrainer",
    "VehicleDataset",
    "VehicleClassifier",
    "LabelSmoothingLoss",
    "MetricTracker",
    "compute_metrics",
    "ModelCheckpoint",
    "TrainingCriteria",
    "TrainingAnalyzer",
]
