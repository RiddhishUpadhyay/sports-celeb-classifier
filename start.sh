#!/bin/bash

echo "Starting FastAPI..."
uvicorn app:app --host 0.0.0.0 --port 8000 &

echo "Waiting for FastAPI..."
sleep 5

echo "Starting Streamlit..."
streamlit run streamlit_app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.enableXsrfProtection=false \
    --server.enableCORS=false \
    --server.headless=true