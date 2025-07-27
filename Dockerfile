# Base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies (for Whisper)
RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    apt-get clean

COPY .env .env
# Copy files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn faster-whisper

# Expose FastAPI port
EXPOSE 8002

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
