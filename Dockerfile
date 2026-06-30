FROM python:3.13-slim

WORKDIR /app

# Install system libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better build caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Hugging Face exposes port 7860
EXPOSE 7860

CMD ["./start.sh"]