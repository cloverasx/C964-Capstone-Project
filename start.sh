#!/bin/bash

# Start nginx in the background
nginx

# Show nginx error log
tail -f /var/log/nginx/error.log &

# Start uvicorn
uvicorn backend.src.api.vehicle_identification_api:app --host 0.0.0.0 --port 8000 --reload 