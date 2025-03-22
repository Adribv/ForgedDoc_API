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
    tesseract --version

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MALLOC_ARENA_MAX=2
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV TMPDIR=/tmp

# Create and set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temporary directory
RUN mkdir -p /tmp/pdf_processing && \
    chmod 777 /tmp/pdf_processing

# Test installations
RUN python -c "from pdf2image import convert_from_path; from PIL import Image; import pytesseract; print('Dependencies verified')"

# Set the command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", \
     "--timeout", "120", \
     "--workers", "2", \
     "--threads", "4", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--log-level", "info", \
     "--preload", \
     "app:app"] 