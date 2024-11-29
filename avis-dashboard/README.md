# Automated Vehicle Identification System

A web-based application that uses deep learning to identify vehicle make, model, and year from images. The system provides a user-friendly interface for uploading images or providing image URLs and displays detailed predictions with confidence scores.

## Features

- **Image Upload**: Drag-and-drop or click to upload vehicle images
- **URL Support**: Analyze vehicles from image URLs
- **Real-time Analysis**: Instant predictions using a deep learning model
- **Detailed Results**: Shows top predictions with confidence scores
- **Batch Processing**: Support for analyzing multiple images
- **Responsive Design**: Works on desktop and mobile devices

## Technology Stack

### Frontend (avis-dashboard)
- React.js
- Tailwind CSS
- Lucide Icons
- Modern JavaScript (ES6+)

### Backend
- FastAPI
- PyTorch
- NumPy
- Pillow

### ML Pipeline
- PyTorch Lightning
- Pandas
- torchvision
- ConvNeXT architecture

## Project Structure

```
project/
├── avis-dashboard/        # React frontend
│   ├── src/
│   │   ├── AVISDashboard.js
│   │   ├── App.js
│   │   └── App.test.js
│   └── public/
├── backend/              # FastAPI service
│   └── src/
│       └── api/
├── ml/                  # ML components
│   ├── models/         # Model inference
│   │   ├── model_inference.py
│   │   └── test_model.py
│   ├── train/         # Training pipeline
│   │   ├── train.py
│   │   └── training_pipeline/
│   │       ├── dataset.py
│   │       ├── trainer.py
│   │       └── training_utils.py
│   └── pretrain/      # Pretraining utilities
├── requirements.txt    # Project dependencies
└── requirements.in    # Core dependencies (for pip-compile)
```

## Getting Started

### Prerequisites
- Node.js (v14 or later)
- Python 3.8 or later
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd avis-dashboard
```

2. Install frontend dependencies:
```bash
npm install
```

3. Install backend dependencies:
```bash
pip install -r requirements.txt
```

4. Set up model files:
Place the model files in the following locations:
- `models/stanford_vehicle_classifier/best_model.pt` - Trained model weights
- `models/stanford_vehicle_classifier/label_encoders.pt` - Label encoders

### Running the Application

1. Start the frontend development server:
```bash
npm start
```

2. Start the backend server:
```bash
uvicorn api.vehicle_identification_api:app --reload --log-level warning
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## API Documentation

### POST /api/predict
Generates predictions for a vehicle image.

#### Request
- Multipart form data with either:
  - `image`: Image file upload
  - `image_url`: URL of an image

#### Response
```json
{
  "make": {
    "prediction": "string",
    "confidence": float,
    "top_5": [["string", float]]
  },
  "model": {
    "prediction": "string",
    "confidence": float,
    "top_5": [["string", float]]
  },
  "year": {
    "prediction": "string",
    "confidence": float,
    "top_5": [["string", float]]
  }
}
```

## Development

### Code Style
- Frontend: ESLint with React recommended rules
- Backend: PEP 8 Python style guide

### Testing
- Frontend: Jest for component and integration testing
- Backend: pytest for API testing
- Test coverage for core UI components and API endpoints

### Model Testing
The project includes a separate testing utility (`test_model.py`) for:
- Batch testing on multiple images
- Model accuracy evaluation
- Visualization of predictions

### Model Training

The project includes a complete training pipeline for the vehicle classification model:

1. Data Preparation:
```bash
python ml/train/prepare_data.py --data_dir /path/to/stanford_cars
```

2. Model Training:
```bash
python ml/train/train.py
```

Training configuration can be modified in `ml/train/train.py`, including:
- Model architecture (ConvNeXT backbone)
- Training parameters (batch size, learning rate, etc.)
- Data augmentation settings
- Early stopping criteria

### Batch Processing

The system supports batch processing of multiple images:

```python
from ml.models.test_model import batch_process_images

results = batch_process_images(
    image_dir="path/to/images",
    output_file="results.csv"
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Vehicle classification model trained on the Stanford Cars Dataset
- UI design inspired by modern web applications
- Thanks to all contributors and maintainers
