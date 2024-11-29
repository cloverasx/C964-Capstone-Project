"""
Training script for the Vehicle Identification System.

This script serves as the entry point for training the vehicle classification model.
It configures and initializes the training process using the components from the
training pipeline.
"""

import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from ml.train.training_pipeline.trainer import VehicleTrainer  # noqa: E402


def main():
    # Training configuration
    config = {
        # Model configuration
        "backbone": "convnext_base",
        "pretrained": True,
        # Training parameters
        "batch_size": 16,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 0.05,
        "label_smoothing": 0.1,
        # Training optimizations
        "mixed_precision": True,
        "gradient_clip": 1.0,
        # Learning rate scheduling
        "scheduler": "one_cycle",
        "warmup_epochs": 5,
        # Early stopping criteria
        "min_epochs": 15,
        "max_epochs": 100,
        "patience": 15,
        "min_val_acc": 0.90,
        # Augmentation settings
        "augment_prob": 0.8,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 0.2,
    }

    # Set up paths
    data_dir = Path("C:/Users/awayn/programming/data/processed/stanford_cars_processed")
    output_dir = ROOT_DIR / "ml" / "models" / "vehicle_classifier"

    # Verify data files exist
    required_files = [
        "train_split.csv",
        "val_split.csv",
        "processed_images.pt",
        "label_encoders.pt",
    ]

    for file in required_files:
        file_path = data_dir / file
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}\n"
                "Please run the data preparation script first."
            )

    # Initialize trainer
    trainer = VehicleTrainer(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        config=config,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
