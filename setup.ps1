# Setup script for AVIS (Automated Vehicle Identification System)
Write-Host "Starting AVIS setup..." -ForegroundColor Green

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow
$prerequisites = @("docker", "docker-compose", "git")
$missing = @()

foreach ($cmd in $prerequisites) {
    if (-not (Test-Command $cmd)) {
        $missing += $cmd
    }
}

if ($missing.Count -gt 0) {
    Write-Host "Error: Missing required tools: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "Please install the missing tools and try again." -ForegroundColor Red
    exit 1
}

# Create necessary directories
Write-Host "Creating directories..." -ForegroundColor Yellow
$modelDir = "ml/models/vehicle_classifier"
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

# Download model files
Write-Host "Downloading model files..." -ForegroundColor Yellow
$releaseUrl = "https://github.com/cloverasx/C964-Capstone-Project/releases/latest/download/model_files.zip"
$zipPath = "model_files.zip"

try {
    Invoke-WebRequest -Uri $releaseUrl -OutFile $zipPath
    Write-Host "Model files downloaded successfully." -ForegroundColor Green
} catch {
    Write-Host "Error downloading model files: $_" -ForegroundColor Red
    exit 1
}

# Extract model files
Write-Host "Extracting model files..." -ForegroundColor Yellow
try {
    Expand-Archive -Path $zipPath -DestinationPath "." -Force
    Remove-Item $zipPath
    Write-Host "Model files extracted successfully." -ForegroundColor Green
} catch {
    Write-Host "Error extracting model files: $_" -ForegroundColor Red
    exit 1
}

# Start Docker containers
Write-Host "Starting Docker containers..." -ForegroundColor Yellow
try {
    docker-compose up --build -d
    Write-Host "Docker containers started successfully." -ForegroundColor Green
} catch {
    Write-Host "Error starting Docker containers: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`nSetup completed successfully!" -ForegroundColor Green
Write-Host "You can access the application at: http://localhost" -ForegroundColor Cyan
Write-Host "API endpoints are available at: http://localhost:8000" -ForegroundColor Cyan 