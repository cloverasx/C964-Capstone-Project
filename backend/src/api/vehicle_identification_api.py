"""
FastAPI Backend for Vehicle Identification System

This module provides the REST API endpoints for the Vehicle Identification System.
It handles image uploads and URL-based image analysis, using a pre-trained deep
learning model to identify vehicle make, model, and year.

API Endpoints:
    POST /api/predict: Accepts an image file or URL and returns vehicle predictions
    GET /api/vehicles: Returns a list of vehicles for recommendations
"""

from fastapi import FastAPI, UploadFile, Form, HTTPException, Response, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Set
import re
import json

# Get the logger but don't configure it - let uvicorn handle the configuration
logger = logging.getLogger("avis")


def convert_numpy_type(obj):
    """
    Convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.intc, np.intp)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_type(x) for x in obj)
    return obj


def normalize_string(s):
    """Normalize string for comparison by removing all non-alphanumeric characters and converting to lowercase."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def serialize_predictions(predictions):
    """
    Convert model predictions to JSON-serializable format and ensure predictions are valid combinations.
    """
    if not predictions:
        logger.error("Predictions is None or empty")
        return None

    try:
        result = {}
        # First, get the raw predictions
        for key in ["make", "model", "year"]:
            if key in predictions:
                pred_dict = predictions[key]
                result[key] = {
                    "prediction": str(convert_numpy_type(pred_dict["prediction"])),
                    "confidence": float(convert_numpy_type(pred_dict["confidence"])),
                    "top_5": [
                        (
                            str(convert_numpy_type(label)),
                            float(convert_numpy_type(conf)),
                        )
                        for label, conf in pred_dict["top_5"]
                    ],
                }
            else:
                logger.warning(f"Missing {key} in predictions")

        # Get all valid combinations
        valid_combinations = get_valid_combinations()

        # Create normalized valid combinations for comparison
        normalized_combinations = {
            (
                normalize_string(combo["make"]),
                normalize_string(combo["model"]),
                combo["year"],
            )
            for combo in valid_combinations
        }

        # Normalize the main prediction
        main_prediction = (
            normalize_string(result["make"]["prediction"]),
            normalize_string(result["model"]["prediction"]),
            int(result["year"]["prediction"]),
        )

        # Debug log
        logger.info(f"Checking normalized prediction: {main_prediction}")
        logger.info(
            f"Sample normalized valid combinations: {list(normalized_combinations)[:5]}"
        )

        # Check if the predicted combination exists in valid combinations
        valid_prediction = main_prediction in normalized_combinations

        if not valid_prediction:
            logger.warning(
                f"Note: Prediction {main_prediction} not found in exact form in dataset"
            )
            logger.info("Keeping original predictions as they may be valid variations")

        return result
    except Exception as e:
        logger.error(f"Error during serialization: {str(e)}", exc_info=True)
        return None


# Set up paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Add project root to Python path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    logger.info(f"Added {ROOT_DIR} to Python path")

# Import model after updating Python path
try:
    from ml.models.model_inference import ModelInference

    logger.info("Successfully imported required modules")
except ImportError as e:
    logger.error(f"Import Error: {e}")
    logger.error(f"Current directory: {os.getcwd()}")
    logger.error(f"Python path: {sys.path}")
    raise

# Load vehicle data from JSON file
VEHICLE_DATA_PATH = Path(__file__).parent / "vehicle_data.json"
with open(VEHICLE_DATA_PATH) as f:
    VALID_VEHICLES = json.load(f)


def get_valid_combinations():
    """Returns a list of all valid make/model/year/body_style combinations."""
    combinations = []
    for make in VALID_VEHICLES:
        for model in make["models"]:
            for year in model["years"]:
                for body_style in model["body_styles"]:
                    combinations.append(
                        {
                            "make": make["make"],
                            "model": model["name"],
                            "year": year,
                            "body_style": body_style,
                        }
                    )
    return combinations


def validate_vehicle_input(make, model, year, body_style):
    """Validates if the input combination exists in our dataset."""
    valid_combinations = get_valid_combinations()

    # Normalize input
    make = make.strip()
    model = model.strip()
    try:
        year = int(year)
    except (ValueError, TypeError):
        return False
    body_style = body_style.strip()

    # Check if combination exists
    for combo in valid_combinations:
        if (
            combo["make"].lower() == make.lower()
            and combo["model"].lower() == model.lower()
            and combo["year"] == year
            and combo["body_style"].lower() == body_style.lower()
        ):
            return True

    return False


# Initialize FastAPI app
app = FastAPI(
    title="Vehicle Identification API",
    description="API for identifying vehicle make, model, and year from images",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:80",
        "http://127.0.0.1:80",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Model paths
MODEL_PATH = str(ROOT_DIR / "ml" / "models" / "vehicle_classifier" / "best_model.pt")
ENCODERS_PATH = str(
    ROOT_DIR / "ml" / "models" / "vehicle_classifier" / "label_encoders.pt"
)

# Verify model files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
if not os.path.exists(ENCODERS_PATH):
    raise FileNotFoundError(f"Encoders file not found at {ENCODERS_PATH}")

# Initialize model
try:
    model = ModelInference(
        model_path=MODEL_PATH,
        label_encoders_path=ENCODERS_PATH,
    )
except Exception as e:
    logger.error(f"Error initializing ModelInference: {e}")
    logger.error(f"Model path: {MODEL_PATH}")
    logger.error(f"Encoders path: {ENCODERS_PATH}")
    raise


@app.post("/api/predict")
async def predict(image: UploadFile = None, image_url: str = Form(None)):
    """
    Generate predictions for a vehicle image.

    Args:
        image: Optional uploaded image file
        image_url: Optional image URL

    Returns:
        Dictionary containing predictions for make, model, and year

    Raises:
        HTTPException: If the request is invalid or processing fails
    """
    try:
        if not image and not image_url:
            raise HTTPException(status_code=400, detail="No image or URL provided")

        if image:
            temp_path = f"temp_{image.filename}"
            try:
                with open(temp_path, "wb") as f:
                    content = await image.read()
                    if not content:
                        raise HTTPException(
                            status_code=400, detail="Empty file uploaded"
                        )
                    f.write(content)
                predictions = model.predict_single(temp_path, visualize=False)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            predictions = model.predict_single(image_url, visualize=False)

        if predictions is None:
            logger.error("Model returned None predictions")
            raise HTTPException(
                status_code=500, detail="Failed to generate predictions"
            )

        serialized = serialize_predictions(predictions)
        if serialized is None:
            logger.error("Failed to serialize predictions")
            raise HTTPException(
                status_code=500, detail="Failed to serialize predictions"
            )

        return serialized

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/vehicles")
async def get_vehicles() -> List[Dict[str, Any]]:
    """
    Get the list of vehicles from the vehicle database.
    Returns a list of valid vehicle combinations.
    """
    logger.info("Received request for vehicle list")

    try:
        combinations = get_valid_combinations()
        if not combinations:
            logger.error("No valid vehicle combinations available")
            raise HTTPException(
                status_code=500, detail="Vehicle database not available"
            )

        logger.info(f"Returning {len(combinations)} vehicle combinations")
        return combinations
    except Exception as e:
        logger.error(f"Error getting vehicle database: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend")
async def recommend_vehicles(request: Request):
    try:
        data = await request.json()
        make = data.get("make", "").strip()
        model = data.get("model", "").strip()
        year = int(data.get("year", 0))
        body_style = data.get("body_style", "").strip()

        # Validate input combination
        if not validate_vehicle_input(make, model, year, body_style):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid vehicle combination. Please check make, model, year, and body style."
                },
            )

        # Get all valid combinations for recommendations
        all_combinations = get_valid_combinations()

        # Find similar vehicles (same make or same body style), excluding the exact match
        similar_vehicles = [
            combo
            for combo in all_combinations
            if
            (
                # Must share either make or body style
                (
                    combo["make"].lower() == make.lower()
                    or combo["body_style"].lower() == body_style.lower()
                )
                # But must not be the exact same vehicle
                and not (
                    combo["make"].lower() == make.lower()
                    and combo["model"].lower() == model.lower()
                    and combo["year"] == year
                    and combo["body_style"].lower() == body_style.lower()
                )
            )
        ]

        # Sort by relevance (same make first, then same body style)
        similar_vehicles.sort(
            key=lambda x: (
                x["make"].lower() == make.lower(),
                x["body_style"].lower() == body_style.lower(),
            ),
            reverse=True,
        )

        # Take top 5 recommendations
        recommendations = similar_vehicles[:5]

        return JSONResponse(content={"recommendations": recommendations})

    except Exception as e:
        logger.error(f"Error in recommend_vehicles: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
