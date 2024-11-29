import torch
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import logging


class DataPreparator:
    def __init__(self, train_dir, test_dir, output_dir):
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup transforms
        self.transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger("DataPreparator")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(self.output_dir / "prepare_data.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def parse_image_path(self, path: Path, dataset_type: str):
        """Parse Stanford Cars dataset path format"""
        try:
            base_dir = self.train_dir if dataset_type == "train" else self.test_dir
            rel_path = str(path.relative_to(base_dir))
            dir_name = path.parent.name

            # Extract year and name
            year = int(dir_name.split()[-1])
            name_without_year = " ".join(dir_name.split()[:-1])

            # Extract make and model
            make = name_without_year.split()[0]
            model = " ".join(name_without_year.split()[1:])

            return {
                "image_path": str(path),
                "relative_path": rel_path,
                "make": make.lower(),
                "model": model.lower(),
                "year": year,
                "split": dataset_type,
            }
        except Exception as e:
            self.logger.error(f"Error parsing path {path}: {str(e)}")
            return None

    def create_metadata(self):
        """Create dataset metadata for both train and test sets"""
        self.logger.info("Creating metadata...")
        records = []

        # Process train directory
        self.logger.info("Processing train directory...")
        train_files = list(self.train_dir.rglob("*.jpg")) + list(
            self.train_dir.rglob("*.jpeg")
        )
        for image_path in tqdm(train_files, desc="Processing train images"):
            metadata = self.parse_image_path(image_path, "train")
            if metadata:
                records.append(metadata)

        # Process test directory
        self.logger.info("Processing test directory...")
        test_files = list(self.test_dir.rglob("*.jpg")) + list(
            self.test_dir.rglob("*.jpeg")
        )
        for image_path in tqdm(test_files, desc="Processing test images"):
            metadata = self.parse_image_path(image_path, "test")
            if metadata:
                records.append(metadata)

        df = pd.DataFrame(records)

        # Create label encoders
        self.logger.info("Creating label encoders...")
        encoders = {}
        for col in ["make", "model", "year"]:
            encoder = LabelEncoder()
            df[f"{col}_encoded"] = encoder.fit_transform(df[col])
            encoders[f"{col}_encoder"] = encoder

            # Log unique values
            self.logger.info(f"Number of unique {col}s: {len(encoder.classes_)}")

        # Save encoders
        torch.save(encoders, self.output_dir / "label_encoders.pt")

        return df

    def process_images(self, metadata_df):
        """Process images and create tensor dict"""
        self.logger.info("Processing images...")
        processed_images = {}

        for _, row in tqdm(
            metadata_df.iterrows(), total=len(metadata_df), desc="Processing"
        ):
            try:
                # Load and transform image
                image = Image.open(row["image_path"]).convert("RGB")
                tensor = self.transform(image)

                # Store using relative path as key
                processed_images[row["relative_path"]] = tensor

            except Exception as e:
                self.logger.error(
                    f"Error processing image {row['image_path']}: {str(e)}"
                )
                continue

        # Save processed images
        torch.save(processed_images, self.output_dir / "processed_images.pt")
        return processed_images

    def save_splits(self, df):
        """Save train and test splits"""
        self.logger.info("Saving dataset splits...")

        # Split based on 'split' column
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "test"]  # Using test set as validation

        # Save splits
        train_df.to_csv(self.output_dir / "train_split.csv", index=False)
        val_df.to_csv(self.output_dir / "val_split.csv", index=False)

        self.logger.info(f"Train set: {len(train_df)} images")
        self.logger.info(f"Validation set: {len(val_df)} images")

        return train_df, val_df

    def prepare_data(self):
        """Execute complete preparation pipeline"""
        try:
            # Create metadata
            metadata_df = self.create_metadata()
            self.logger.info(f"Created metadata for {len(metadata_df)} images")

            # Process images
            processed_images = self.process_images(metadata_df)
            self.logger.info(f"Processed {len(processed_images)} images")

            # Save splits
            train_df, val_df = self.save_splits(metadata_df)

            # Save full metadata
            metadata_df.to_csv(self.output_dir / "metadata.csv", index=False)

            self.logger.info("Data preparation completed successfully!")
            return True

        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            return False


def main():
    # Configuration
    train_dir = "C:/Users/awayn/programming/data/stanford_cars/train"
    test_dir = "C:/Users/awayn/programming/data/stanford_cars/test"
    output_dir = "C:/Users/awayn/programming/data/processed/stanford_cars_processed"

    # Initialize and run pipeline
    preparator = DataPreparator(train_dir, test_dir, output_dir)
    success = preparator.prepare_data()

    if success:
        print("Data preparation completed successfully!")
    else:
        print("Data preparation failed. Check logs for details.")


if __name__ == "__main__":
    main()
