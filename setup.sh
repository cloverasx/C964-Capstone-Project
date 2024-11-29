#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting AVIS setup...${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"
prerequisites=("docker" "docker-compose" "git" "curl")
missing=()

for cmd in "${prerequisites[@]}"; do
    if ! command_exists "$cmd"; then
        missing+=("$cmd")
    fi
done

if [ ${#missing[@]} -ne 0 ]; then
    echo -e "${RED}Error: Missing required tools: ${missing[*]}${NC}"
    echo -e "${RED}Please install the missing tools and try again.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p ml/models/vehicle_classifier

# Download model files
echo -e "${YELLOW}Downloading model files...${NC}"
RELEASE_URL="https://github.com/cloverasx/C964-Capstone-Project/releases/latest/download/model_files.zip"
if ! curl -L -o model_files.zip "$RELEASE_URL"; then
    echo -e "${RED}Error downloading model files${NC}"
    exit 1
fi
echo -e "${GREEN}Model files downloaded successfully.${NC}"

# Extract model files
echo -e "${YELLOW}Extracting model files...${NC}"
if ! unzip -o model_files.zip; then
    echo -e "${RED}Error extracting model files${NC}"
    rm model_files.zip
    exit 1
fi
rm model_files.zip
echo -e "${GREEN}Model files extracted successfully.${NC}"

# Start Docker containers
echo -e "${YELLOW}Starting Docker containers...${NC}"
if ! docker-compose up --build -d; then
    echo -e "${RED}Error starting Docker containers${NC}"
    exit 1
fi
echo -e "${GREEN}Docker containers started successfully.${NC}"

echo -e "\n${GREEN}Setup completed successfully!${NC}"
echo -e "${CYAN}You can access the application at: http://localhost${NC}"
echo -e "${CYAN}API endpoints are available at: http://localhost:8000${NC}" 