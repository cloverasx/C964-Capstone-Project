# Automated Vehicle Identification System (AVIS)

AVIS is a web application that uses AI to identify vehicles from images and provide similar vehicle recommendations.

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

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Start the application:
```bash
docker-compose up --build
```

3. Access the application:
- Open your browser and navigate to `http://localhost`
- The backend API will be available at `http://localhost:8000`

## Project Structure

```
.
├── avis-dashboard/        # React frontend
├── backend/              # Python backend
│   └── src/             # Source code
│       └── api/         # API endpoints
├── ml/                  # Machine learning code
│   ├── models/         # Trained models
│   ├── train/         # Training scripts
│   └── pretrain/      # Pre-training utilities
├── requirements.txt     # Python dependencies
├── yolov8n.pt          # YOLOv8 model file
├── Dockerfile.frontend  # Frontend Docker configuration
├── Dockerfile.backend   # Backend Docker configuration
├── docker-compose.yml   # Docker Compose configuration
└── nginx.conf          # Nginx configuration
```

## Development

To run the application in development mode:

1. Frontend:
```bash
cd avis-dashboard
npm install
npm start
```

2. Backend:
```bash
pip install -r requirements.txt
uvicorn backend.src.api.vehicle_identification_api:app --reload
```

## Environment Setup

The application uses Python 3.9 and Node.js 18. All Python dependencies are listed in `requirements.txt` and will be installed automatically when building the Docker containers.

## Docker Configuration

The application is containerized using Docker with two services:
- Frontend: React application served through Nginx
- Backend: FastAPI application with ML capabilities

The containers are configured to work together through Docker Compose, with the frontend proxying API requests to the backend service.

## License

[MIT License](LICENSE) 