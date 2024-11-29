# Stage 1: Build the React frontend
FROM node:18-alpine as frontend-build

WORKDIR /app/frontend
COPY avis-dashboard/package*.json ./
RUN npm install

COPY avis-dashboard/ ./
RUN npm run build

# Stage 2: Build the Python backend and serve the application
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Set up nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=frontend-build /app/frontend/build /usr/share/nginx/html

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install pip-tools
RUN pip install --no-cache-dir --upgrade pip pip-tools setuptools wheel

# Copy and install Python requirements
COPY requirements.in .
RUN pip-compile requirements.in
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend and ML directories
COPY backend /app/backend
COPY ml /app/ml
COPY yolov8n.pt /app/yolov8n.pt

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CORS_ORIGINS="http://localhost,http://localhost:80,http://127.0.0.1,http://127.0.0.1:80"

# Copy the startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 80 8000

# Start both nginx and uvicorn
CMD ["/start.sh"] 