import json
from pathlib import Path
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get dataset path from environment variable or use default
DEFAULT_DATASET_PATH = "C:/Users/awayn/programming/data/stanford_cars"
DATASET_PATH = os.getenv("STANFORD_CARS_DATASET", DEFAULT_DATASET_PATH)


def parse_directory_name(dir_name: str) -> tuple:
    """Parse directory name into make, model, year, and body style."""
    try:
        # Directory format example: "Acura Integra Type R 2001"
        parts = dir_name.split()
        year = int(parts[-1])  # Last part is year
        make = parts[0]  # First part is make

        # Everything between make and year is the model
        model = " ".join(parts[1:-1])

        # Determine body style from the model name or directory name
        body_style = None
        style_keywords = {
            "Sedan": ["Sedan", "SRT-8", "Type-S", "Type R", "IPL"],
            "SUV": ["SUV", "Hybrid SUV"],
            "Coupe": ["Coupe"],
            "Convertible": ["Convertible", "Spyder", "Roadster"],
            "Hatchback": ["Hatchback"],
            "Wagon": ["Wagon"],
            "Van": ["Van"],
            "Minivan": ["Minivan"],
            "Crew Cab": ["Crew Cab", "SUT"],
            "Extended Cab": ["Extended Cab"],
            "Regular Cab": ["Regular Cab"],
        }

        # Check for body style keywords in the model name
        for style, keywords in style_keywords.items():
            if any(keyword in dir_name for keyword in keywords):
                body_style = style
                break

        # Default to Sedan if no body style is found
        if not body_style:
            body_style = "Sedan"

        return make, model, year, body_style
    except Exception as e:
        logger.warning(f"Could not parse directory name: {dir_name} - {e}")
        return None


def build_vehicle_database():
    """Build a hierarchical database of vehicles from the Stanford Cars dataset."""
    try:
        # Path to Stanford Cars dataset
        dataset_path = Path(DATASET_PATH)
        train_path = dataset_path / "train"
        test_path = dataset_path / "test"

        logger.info(f"Looking for vehicle data in: {dataset_path}")

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # Dictionary to store makes and their models
        vehicle_db = {}

        # Process both train and test directories
        for data_path in [train_path, test_path]:
            if not data_path.exists():
                logger.warning(f"Path does not exist: {data_path}")
                continue

            logger.info(f"Processing directory: {data_path}")

            # Get all subdirectories (each represents a vehicle class)
            for vehicle_dir in data_path.iterdir():
                if vehicle_dir.is_dir():
                    parsed = parse_directory_name(vehicle_dir.name)
                    if parsed:
                        make, model, year, body_style = parsed

                        # Initialize make if not exists
                        if make not in vehicle_db:
                            vehicle_db[make] = {"models": {}}

                        # Initialize model if not exists
                        if model not in vehicle_db[make]["models"]:
                            vehicle_db[make]["models"][model] = {
                                "years": set(),
                                "body_styles": set(),
                            }

                        # Add year and body style
                        vehicle_db[make]["models"][model]["years"].add(year)
                        vehicle_db[make]["models"][model]["body_styles"].add(body_style)

        if not vehicle_db:
            raise ValueError("No vehicle data was parsed from the dataset")

        # Convert to the final format
        final_db = []
        for make, make_data in vehicle_db.items():
            make_entry = {
                "make": make,
                "models": [],
            }

            for model, model_data in make_data["models"].items():
                model_entry = {
                    "name": model,
                    "years": sorted(list(model_data["years"])),
                    "body_styles": sorted(list(model_data["body_styles"])),
                }
                make_entry["models"].append(model_entry)

            final_db.append(make_entry)

        # Sort makes alphabetically
        final_db.sort(key=lambda x: x["make"])

        # Save to JSON file
        output_path = Path(__file__).parent / "vehicle_data.json"
        with open(output_path, "w") as f:
            json.dump(final_db, f, indent=2)

        logger.info(f"Successfully saved vehicle database to {output_path}")
        logger.info(f"Found {len(final_db)} makes")
        total_models = sum(len(make["models"]) for make in final_db)
        logger.info(f"Found {total_models} total models")

        # Print some sample data for verification
        logger.info("\nSample data:")
        for make in final_db[:3]:
            logger.info(f"\nMake: {make['make']}")
            for model in make["models"][:2]:
                logger.info(f"  Model: {model['name']}")
                logger.info(f"    Years: {model['years']}")
                logger.info(f"    Body Styles: {model['body_styles']}")

    except Exception as e:
        logger.error(f"Error building vehicle database: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    build_vehicle_database()
