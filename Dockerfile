FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libpoppler-dev \
    libpoppler-cpp-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Verify installations
RUN pdftoppm -v && \
    tesseract --version && \
    ls -l /usr/share/tesseract-ocr/*/tessdata/eng.traineddata

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MALLOC_ARENA_MAX=2
# Set Tesseract data path - will be updated during runtime
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV TMPDIR=/tmp
# Limit Python memory usage
ENV PYTHONMALLOC=malloc
ENV PYTHONUTF8=1
ENV PYTHONHASHSEED=random

# Create and set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temporary directory with proper permissions
RUN mkdir -p /tmp/pdf_processing && \
    chmod 777 /tmp/pdf_processing && \
    mkdir -p /tmp/tesseract_cache && \
    chmod 777 /tmp/tesseract_cache

# Find Tesseract data directory and verify installation
RUN TESSDATA_DIR=$(find /usr/share/tesseract-ocr -name "tessdata" -type d | head -n 1) && \
    echo "Found Tesseract data directory: $TESSDATA_DIR" && \
    python -c "\
from pdf2image import convert_from_path; \
from PIL import Image; \
import pytesseract; \
print('Testing Tesseract...'); \
version = pytesseract.get_tesseract_version(); \
print(f'Tesseract version: {version}'); \
print('Dependencies verified')"

# Create a shell script to run the application
RUN echo '#!/bin/bash\n\
# Find and set Tesseract data directory\n\
export TESSDATA_PREFIX=$(find /usr/share/tesseract-ocr -name "tessdata" -type d | head -n 1)\n\
echo "Using Tesseract data directory: $TESSDATA_PREFIX"\n\
export TESSERACT_CACHE_DIR=/tmp/tesseract_cache\n\
\n\
# Start Gunicorn\n\
gunicorn --bind "0.0.0.0:${PORT:-8000}" \
    --timeout 180 \
    --workers 1 \
    --threads 2 \
    --max-requests 100 \
    --max-requests-jitter 50 \
    --log-level info \
    app:app' > /app/start.sh && \
    chmod +x /app/start.sh

# Set the command to run the application
CMD ["/app/start.sh"] 