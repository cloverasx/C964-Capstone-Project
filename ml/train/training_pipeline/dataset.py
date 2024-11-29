import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import logging


class VehicleDataset(Dataset):
    def __init__(self, csv_file, processed_images_file, transform=None, augment=False):
        print(f"\nDebug: Initializing VehicleDataset")
        print(f"Debug: Loading CSV from {csv_file}")
        self.data = pd.read_csv(csv_file)
        print(f"Debug: Initial CSV rows: {len(self.data)}")

        print(f"Debug: Loading processed images from {processed_images_file}")
        self.processed_images = torch.load(processed_images_file)
        print(f"Debug: Number of processed images: {len(self.processed_images)}")

        # Debug: Print first few paths before conversion
        print("\nDebug: First few relative paths from CSV:")
        print(self.data["relative_path"].head())
        print("\nDebug: First few keys from processed images:")
        print(list(self.processed_images.keys())[:5])

        # Normalize paths to use forward slashes
        processed_paths = {
            path.replace("\\", "/"): tensor
            for path, tensor in self.processed_images.items()
        }

        # Convert CSV paths to use forward slashes
        self.data["normalized_path"] = self.data["relative_path"].str.replace("\\", "/")

        # Filter data to only include images we have processed
        self.data = self.data[
            self.data["normalized_path"].isin(processed_paths.keys())
        ].reset_index(drop=True)

        # Update processed_images to use normalized paths
        self.processed_images = processed_paths

        print(f"\nDebug: After filtering, remaining rows: {len(self.data)}")
        if len(self.data) == 0:
            print("\nDebug: No matching paths found! Checking formats:")
            if len(self.data) > 0:  # Only try to print if we have data
                csv_path = self.data["normalized_path"].iloc[0]
                print(f"CSV path format example: {csv_path}")
            print(f"Processed images key example: {list(processed_paths.keys())[0]}")

        self.transform = transform
        self.augment = augment
        self.target_size = (224, 224)

        logging.info(f"Loaded dataset with {len(self.data)} samples")
        logging.info(f"CSV file: {csv_file}")
        logging.info(f"Number of processed images: {len(self.processed_images)}")

        if augment:
            self.aug_transform = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomAffine(
                        degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
                    ),
                    T.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
                ]
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["normalized_path"]  # Use normalized path

        try:
            image_tensor = self.processed_images[image_path]
        except KeyError:
            raise KeyError(f"Image path not found in processed images: {image_path}")

        if self.augment:
            image = T.ToPILImage()(image_tensor)
            image = self.aug_transform(image)
            image_tensor = T.ToTensor()(image)

        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0),
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        image_tensor = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(image_tensor)

        labels = {
            "make": torch.tensor(row["make_encoded"], dtype=torch.long),
            "model": torch.tensor(row["model_encoded"], dtype=torch.long),
            "year": torch.tensor(row["year_encoded"], dtype=torch.long),
        }

        return image_tensor, labels
