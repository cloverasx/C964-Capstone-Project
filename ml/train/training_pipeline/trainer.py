"""
Training orchestration for the Vehicle Identification System.

This module handles the training process, including data loading, model initialization,
optimization, and evaluation. It coordinates all the components needed for training
the vehicle classification model.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import time

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


class VehicleTrainer:
    def __init__(self, data_dir, output_dir, config=None):
        """
        Initialize trainer with configurations

        Args:
            data_dir: Directory containing processed data
            output_dir: Directory to save training outputs
            config: Training configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default configurations
        self.config = {
            "batch_size": 32,
            "num_epochs": 100,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "label_smoothing": 0.1,
            "warmup_epochs": 5,
            "backbone": "convnext_base",
            "mixed_precision": True,
            "gradient_clip": 1.0,
            "scheduler": "cosine",  # or 'plateau'
            "min_epochs": 10,
            "max_epochs": 100,
            "patience": 10,
            "min_val_acc": 0.95,
            "conv_threshold": 0.001,
        }
        if config:
            self.config.update(config)

        # Setup components
        self.setup_components()

        # Initialize metric tracker
        self.metric_tracker = MetricTracker(self.output_dir)

        # Setup model checkpoint
        self.checkpointer = ModelCheckpoint(self.output_dir, mode="max", verbose=True)

        # Initialize training criteria and analyzer
        self.criteria = TrainingCriteria(self.config)
        self.analyzer = TrainingAnalyzer(window_size=5)

    def setup_components(self):
        """Initialize all training components"""
        print("\nDebug: Setting up training components...")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Debug: Using device: {self.device}")

        # Load dataset info
        encoder_path = self.data_dir / "label_encoders.pt"
        print(f"Debug: Loading encoders from: {encoder_path}")
        print(f"Debug: Encoder file exists: {encoder_path.exists()}")

        encoders = torch.load(self.data_dir / "label_encoders.pt")
        print(
            f"Debug: Number of classes - Makes: {len(encoders['make_encoder'].classes_)}, "
            f"Models: {len(encoders['model_encoder'].classes_)}, "
            f"Years: {len(encoders['year_encoder'].classes_)}"
        )

        self.num_makes = len(encoders["make_encoder"].classes_)
        self.num_models = len(encoders["model_encoder"].classes_)
        self.num_years = len(encoders["year_encoder"].classes_)

        # Debug dataset loading
        train_split_path = self.data_dir / "train_split.csv"
        processed_images_path = self.data_dir / "processed_images.pt"

        print(f"\nDebug: Loading datasets...")
        print(f"Train split path exists: {train_split_path.exists()}")
        print(f"Processed images path exists: {processed_images_path.exists()}")

        if train_split_path.exists():
            import pandas as pd

            df = pd.read_csv(train_split_path)
            print(f"Train split contains {len(df)} rows")
            print("First few rows of train_split.csv:")
            print(df.head())

        # Create datasets
        self.train_dataset = VehicleDataset(
            self.data_dir / "train_split.csv",
            self.data_dir / "processed_images.pt",
            augment=True,
        )
        print(f"\nDebug: Train dataset size: {len(self.train_dataset)}")

        self.val_dataset = VehicleDataset(
            self.data_dir / "val_split.csv",
            self.data_dir / "processed_images.pt",
            augment=False,
        )
        print(f"Debug: Validation dataset size: {len(self.val_dataset)}")

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Initialize model
        self.model = VehicleClassifier(
            self.num_makes,
            self.num_models,
            self.num_years,
            backbone=self.config["backbone"],
        ).to(self.device)

        # Loss functions with label smoothing
        self.criterion = {
            "make": LabelSmoothingLoss(self.num_makes, self.config["label_smoothing"]),
            "model": LabelSmoothingLoss(
                self.num_models, self.config["label_smoothing"]
            ),
            "year": LabelSmoothingLoss(self.num_years, self.config["label_smoothing"]),
        }

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        # Learning rate scheduler
        if self.config["scheduler"] == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.1, patience=5, verbose=True
            )

        # Gradient scaler for mixed precision
        self.scaler = GradScaler(enabled=self.config["mixed_precision"])

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_metrics = {"make_acc": 0, "model_acc": 0, "year_acc": 0}

        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            # Move data to device
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            # Forward pass with mixed precision
            with autocast(enabled=self.config["mixed_precision"]):
                outputs = self.model(images)

                # Calculate losses
                losses = {
                    task: self.criterion[task](outputs[task], labels[task])
                    for task in ["make", "model", "year"]
                }
                loss = sum(losses.values())

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config["gradient_clip"] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["gradient_clip"]
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Compute metrics
            metrics = compute_metrics(outputs, labels)
            epoch_loss += loss.item()

            for k, v in metrics.items():
                epoch_metrics[k] += v

            # Update progress bar
            pbar.set_postfix({"loss": loss.item(), "make_acc": metrics["make_acc"]})

        # Compute epoch averages
        num_batches = len(self.train_loader)
        epoch_loss /= num_batches
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        epoch_metrics["loss"] = epoch_loss

        return epoch_metrics

    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        val_metrics = {"make_acc": 0, "model_acc": 0, "year_acc": 0}

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                # Move data to device
                images = images.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}

                # Forward pass
                outputs = self.model(images)

                # Calculate losses
                losses = {
                    task: self.criterion[task](outputs[task], labels[task])
                    for task in ["make", "model", "year"]
                }
                loss = sum(losses.values())

                # Compute metrics
                metrics = compute_metrics(outputs, labels)
                val_loss += loss.item()

                for k, v in metrics.items():
                    val_metrics[k] += v

        # Compute averages
        num_batches = len(self.val_loader)
        val_loss /= num_batches
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        val_metrics["loss"] = val_loss

        return val_metrics

    def train(self):
        """Complete training loop"""
        print(f"Training on {self.device}")
        best_val_acc = 0
        patience_counter = 0
        start_time = time.time()
        prev_val_metrics = None

        for epoch in range(self.config["max_epochs"]):
            print(f"\nEpoch {epoch+1}/{self.config['max_epochs']}")

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.validate()

            # Update analyzer
            self.analyzer.update(val_metrics["loss"], val_metrics["make_acc"])

            # Check for overfitting
            if self.analyzer.is_overfitting():
                print("Warning: Possible overfitting detected!")

            # Update learning rate
            if self.config["scheduler"] == "cosine":
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics["loss"])

            # Update metrics
            self.metric_tracker.update(train_metrics, "train")
            self.metric_tracker.update(val_metrics, "val")

            # Checkpoint saving
            improved = self.checkpointer(
                val_metrics["make_acc"],
                self.model,
                epoch,
                self.optimizer,
                self.scheduler,
            )

            if not improved:
                patience_counter += 1
            else:
                patience_counter = 0

            # Check stopping criteria
            if self.criteria.should_stop(
                epoch, val_metrics, patience_counter, prev_val_metrics
            ):
                print("Training stopped based on criteria!")
                break

            prev_val_metrics = val_metrics.copy()

            # Print epoch summary
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Train Make Acc: {train_metrics['make_acc']:.4f}")
            print(f"Val Make Acc: {val_metrics['make_acc']:.4f}")

        # Training completed
        training_time = (time.time() - start_time) / 60
        print(f"\nTraining completed in {training_time:.2f} minutes")

        # Save final plots
        self.metric_tracker.plot_metrics()

        return self.metric_tracker.history
