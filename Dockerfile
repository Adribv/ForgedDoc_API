# Set up Tesseract paths and verify installation
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
    tesseract --list-langs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MALLOC_ARENA_MAX=1
ENV TESSDATA_PREFIX=/usr/share/tessdata
ENV TESSERACT_PATH=/usr/bin/tesseract
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV TMPDIR=/tmp
# Limit Python memory usage
ENV PYTHONMALLOC=malloc
ENV PYTHONUTF8=1
ENV PYTHONHASHSEED=random
# Limit memory usage
ENV GUNICORN_CMD_ARGS="--preload --max-requests 50 --max-requests-jitter 10 --workers 1 --threads 1 --timeout 300"

# Create and set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temporary directories with proper permissions
RUN mkdir -p /tmp/pdf_processing && \
    chmod 777 /tmp/pdf_processing && \
    mkdir -p /tmp/tesseract_cache && \
    chmod 777 /tmp/tesseract_cache

# Verify Python dependencies and Tesseract
RUN python -c "\
import sys; \
import pytesseract; \
print('Python version:', sys.version); \
print('Tesseract version:', pytesseract.get_tesseract_version()); \
print('Tesseract data prefix:', pytesseract.get_tesseract_prefix()); \
print('Dependencies verified')"

# Create a shell script to run the application
RUN echo '#!/bin/bash\n\
# Verify Tesseract configuration\n\
echo "Tesseract configuration:"\n\
echo "TESSDATA_PREFIX=$TESSDATA_PREFIX"\n\
echo "TESSERACT_PATH=$TESSERACT_PATH"\n\
tesseract --list-langs\n\
\n\
# Clear any temporary files\n\
rm -rf /tmp/pdf_processing/*\n\
rm -rf /tmp/tesseract_cache/*\n\
\n\
# Start Gunicorn with memory optimizations\n\
exec gunicorn \
    --bind "0.0.0.0:${PORT:-8000}" \
    --worker-class=gthread \
    --worker-tmp-dir /dev/shm \
    --log-level info \
    app:app' > /app/start.sh && \
    chmod +x /app/start.sh

# Set the command to run the application
CMD ["/app/start.sh"]