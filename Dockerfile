FROM python:3.9-slim

# Install system dependencies and poppler-utils
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpoppler-dev \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify poppler installation
RUN pdftoppm -v && \
    pdfinfo -v

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV PATH="/usr/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# Create directory for temporary files
RUN mkdir -p /tmp/pdf2image
ENV PDF2IMAGE_TEMP_DIR=/tmp/pdf2image

# Test poppler installation
RUN python -c "from pdf2image import convert_from_path; print('Poppler installation verified')"

# Expose port
EXPOSE 8000

# Run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app 