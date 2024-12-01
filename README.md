# Automated Vehicle Identification System (AVIS)

AVIS is a web application that uses AI to identify vehicles from images and provide similar vehicle recommendations. The system uses deep learning to analyze vehicle images and can identify the make, model, and year of vehicles, while also suggesting similar vehicles based on the identified characteristics.

## Features

- Upload images via drag-and-drop or file selection
- Input image URLs directly
- AI-powered vehicle identification (make, model, year)
- Similar vehicle recommendations
- Confidence scores for predictions
- Interactive tooltips for feature explanations
- Model training metrics visualization
- Modern, responsive UI

## Prerequisites

For Docker installation (recommended):
- Docker Desktop installed and running
- Docker Compose installed
- Git installed
- PowerShell (Windows) or Bash (Linux/Mac)

For local installation:
- Python 3.12
- Node.js 16 or higher
- Git installed
- npm installed

## Quick Start

### Docker Installation (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/cloverasx/C964-Capstone-Project.git
cd C964-Capstone-Project
```

2. Run the setup script:
Windows (PowerShell):
```powershell
.\setup.ps1
```

Linux/Mac (Bash):
```bash
chmod +x setup.sh
./setup.sh
```

3. Access the application:
- Frontend: Open your browser and navigate to `http://localhost`
- Backend API: Available at `http://localhost:8000`

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/cloverasx/C964-Capstone-Project.git
cd C964-Capstone-Project
```

2. Download model files:
   - Go to the [Releases](https://github.com/cloverasx/C964-Capstone-Project/releases) page
   - Download `model_files.zip` from the latest release
   - Extract the contents to their respective directories:
     - Place `best_model.pt` in `ml/models/vehicle_classifier/`
     - Place `label_encoders.pt` in `ml/models/vehicle_classifier/`

3. Set up the backend:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.src.api.vehicle_identification_api:app --reload
```

4. Set up the frontend:
```bash
cd avis-dashboard
npm install
npm start
```

5. Access the application:
- Frontend: Open your browser and navigate to `http://localhost:3000`
- Backend API: Available at `http://localhost:8000`

## Project Structure

```
C964-Capstone-Project/
├── avis-dashboard/                             # Handles frontend UI and interactions
│   ├── src/                                    # Contains frontend source code
│   │   ├── AVISDashboard.js                    # Manages main application state and layout
│   │   ├── VehicleVisualizations.js            # Renders prediction charts and graphs
│   │   ├── index.js                            # Initializes React application
│   │   └── index.css                           # Defines global styles
│   ├── package.json                            # Manages frontend dependencies
│   └── tailwind.config.js                      # Configures Tailwind CSS styling
├── backend/                                    # Handles API requests and ML inference
│   └── src/                                    # Contains backend source code
│       └── api/                                # Defines API endpoints
│           ├── vehicle_identification_api.py   # Processes image uploads and predictions
│           ├── vehicle_data.json               # Stores vehicle database
│           └── generate_vehicle_data.py        # Creates vehicle database from raw data
├── ml/                                         # Handles machine learning operations
│   ├── models/                                 # Manages model files and inference
│   │   ├── model_inference.py                  # Runs predictions on images
│   │   └── vehicle_classifier/                 # Stores trained models
│   │       ├── best_model.pt                   # Performs vehicle classification
│   │       ├── label_encoders.pt               # Maps predictions to labels
│   │       └── accuracy_plots.png              # Displays model training metrics
│   ├── train/                                  # Handles model training
│   │   ├── training_pipeline/                  # Defines training components
│   │   │   ├── model.py                        # Defines neural network architecture
│   │   │   ├── dataset.py                      # Loads and processes training data
│   │   │   └── trainer.py                      # Executes training loop
│   │   └── train.py                            # Runs complete training pipeline
│   └── pretrain/                               # Prepares training data
│       └── preparator.py                       # Processes raw data for training
├── Docker Configuration
│   ├── Dockerfile                              # Builds multi-stage container image
│   ├── docker-compose.yml                      # Orchestrates container services
│   ├── nginx.conf                              # Routes HTTP traffic
│   └── start.sh                                # Initializes container services
└── Configuration Files
    ├── requirements.in                         # Defines Python dependencies - builds requirements.txt
    ├── requirements.txt                        # Lists locked Python dependencies
    ├── setup.ps1                               # Automates Windows setup
    ├── setup.sh                                # Automates Linux/Mac setup
    └── .gitignore                              # Excludes files from version control
```

### Key Components

- **Frontend**: Provides user interface for vehicle identification
- **Backend**: Processes requests and runs ML inference
- **Machine Learning**: Trains and runs vehicle classification
- **Docker Configuration**: Builds and runs containerized services
- **Configuration Files**: Manages dependencies and setup

## Required Model Files

The following model files are required for the application to function:

1. `best_model.pt` (ml/models/vehicle_classifier/)
   - Main vehicle classification model
   - Size: ~1GB

2. `label_encoders.pt` (ml/models/vehicle_classifier/)
   - Label encoders for prediction mapping
   - Size: ~6.2KB

These files can be downloaded from the [Releases](https://github.com/cloverasx/C964-Capstone-Project/releases) page.

## Development Setup

### Frontend Development
```bash
cd avis-dashboard
npm install
npm start
```
The frontend development server will run on `http://localhost:3000`

### Backend Development
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn backend.src.api.vehicle_identification_api:app --reload
```
The backend development server will run on `http://localhost:8000`

## Docker Configuration

The application is containerized using Docker with a multi-stage build process:

1. Frontend Build Stage:
   - Builds the React application
   - Optimizes for production

2. Backend and Server Stage:
   - Sets up Python environment
   - Configures Nginx as reverse proxy
   - Handles both static files and API requests

## API Endpoints

- `POST /api/predict`: Upload image for vehicle identification
- `GET /api/vehicles`: Get list of known vehicles
- `POST /api/recommend`: Get vehicle recommendations

## Environment Variables

The following environment variables can be configured:
- `PYTHONPATH`: Set to /app in container
- `PYTHONUNBUFFERED`: Set to 1 for better logging
- `CORS_ORIGINS`: Allowed origins for CORS

## License

[MIT License](LICENSE) 