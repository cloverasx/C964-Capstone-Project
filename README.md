# Automated Vehicle Identification System (AVIS)

AVIS is a web application that uses AI to identify vehicles from images and provide similar vehicle recommendations. The system uses deep learning to analyze vehicle images and can identify the make, model, and year of vehicles, while also suggesting similar vehicles based on the identified characteristics.

## Features

- Upload images via drag-and-drop or file selection
- Input image URLs directly
- AI-powered vehicle identification (make, model, year)
- Similar vehicle recommendations
- Confidence scores for predictions
- Modern, responsive UI

## Prerequisites

- Docker
- Docker Compose
- Git

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/cloverasx/C964-Capstone-Project.git
cd C964-Capstone-Project
```

2. Start the application:
```bash
docker-compose up --build
```

3. Access the application:
- Frontend: Open your browser and navigate to `http://localhost`
- Backend API: Available at `http://localhost:8000`

## Project Structure

```
.
├── avis-dashboard/          # React frontend
│   ├── public/             # Static files
│   └── src/               # React source code
│       ├── components/    # React components
│       └── features/     # Feature-specific components
├── backend/               # Python backend
│   └── src/              # Source code
│       └── api/          # API endpoints
├── ml/                   # Machine learning code
│   ├── models/          # Model inference and saved models
│   ├── train/          # Training pipeline
│   └── pretrain/       # Data preparation scripts
├── Docker Files
│   ├── Dockerfile       # Main Docker configuration
│   ├── docker-compose.yml # Docker Compose configuration
│   ├── nginx.conf      # Nginx configuration
│   └── start.sh        # Container startup script
└── Configuration Files
    ├── requirements.in  # Python package requirements
    ├── requirements.txt # Locked Python dependencies
    └── .gitignore      # Git ignore rules

```

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