"""
Model inference module for the Vehicle Identification System.

This module provides the ModelInference class which handles loading the trained model
and generating predictions for new images.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import logging
from ml.train.training_pipeline.model import VehicleClassifier
import requests
from io import BytesIO
import re
from pathlib import Path


class ModelInference:
    def __init__(self, model_path, label_encoders_path):
        """
        Initialize model inference.

        Args:
            model_path: Path to saved model checkpoint
            label_encoders_path: Path to saved label encoders
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load label encoders
        self.encoders = torch.load(label_encoders_path)

        # Set up model
        self.setup_model(model_path)

        # Setup transforms
        self.transform = T.Compose(
            [
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Setup logging
        self.logger = logging.getLogger("avis.model")

    def setup_model(self, model_path):
        """Load and setup the model"""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Initialize model
        self.model = VehicleClassifier(
            num_makes=len(self.encoders["make_encoder"].classes_),
            num_models=len(self.encoders["model_encoder"].classes_),
            num_years=len(self.encoders["year_encoder"].classes_),
        ).to(self.device)

        # Load model weights
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

    def is_url(self, path):
        """Check if the path is a URL"""
        path_str = str(path) if isinstance(path, Path) else path
        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain
            r"localhost|"  # localhost
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ip
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",  # path
            re.IGNORECASE,
        )
        return url_pattern.match(path_str) is not None

    def load_image(self, image_path):
        """Load image from file or URL"""
        try:
            self.logger.info(f"Loading image from: {image_path}")
            if self.is_url(image_path):
                self.logger.info("Detected URL, downloading image...")
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                self.logger.info("Image downloaded successfully")
                image = Image.open(BytesIO(response.content))
            else:
                self.logger.info("Loading local file...")
                image_path_str = (
                    str(image_path) if isinstance(image_path, Path) else image_path
                )
                image = Image.open(image_path_str)
                self.logger.info("Local file loaded successfully")

            image = image.convert("RGB")
            self.logger.info(f"Image converted to RGB. Size: {image.size}")
            return image
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}", exc_info=True)
            raise

    def predict_single(self, image_path, visualize=False):
        """
        Make prediction on a single image

        Args:
            image_path: Path to image file or URL
            visualize: Whether to display the image with predictions (unused in API)

        Returns:
            Dictionary containing predictions and confidence scores
        """
        try:
            self.logger.info(f"Starting prediction for {image_path}")

            # Load and process image
            self.logger.info("Loading image...")
            image = self.load_image(image_path)
            self.logger.info("Image loaded successfully")

            self.logger.info("Processing image...")
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            self.logger.info(f"Image processed. Tensor shape: {img_tensor.shape}")

            # Get predictions
            self.logger.info("Running model inference...")
            with torch.no_grad():
                outputs = self.model(img_tensor)
            self.logger.info("Model inference complete")

            # Process predictions
            self.logger.info("Processing predictions...")
            predictions = {}
            for task, encoder in [
                ("make", self.encoders["make_encoder"]),
                ("model", self.encoders["model_encoder"]),
                ("year", self.encoders["year_encoder"]),
            ]:
                logits = outputs[task][0]
                probs = torch.softmax(logits, dim=0)
                conf, idx = torch.max(probs, dim=0)

                # Get top 5 predictions
                top5_probs, top5_idx = torch.topk(probs, 5)
                top5_classes = encoder.inverse_transform(top5_idx.cpu().numpy())
                top5_predictions = list(zip(top5_classes, top5_probs.cpu().numpy()))

                predictions[task] = {
                    "prediction": encoder.inverse_transform([idx.cpu().numpy()])[0],
                    "confidence": conf.item(),
                    "top_5": top5_predictions,
                }
                self.logger.info(
                    f"{task.capitalize()} prediction: {predictions[task]['prediction']} ({predictions[task]['confidence']:.2%})"
                )

            self.logger.info("Prediction complete")
            return predictions

        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}", exc_info=True)
            return None
