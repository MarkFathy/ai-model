FROM python:3.12-slim

# Install system dependencies needed by OpenCV and PaddleOCR (C++ and graphics libraries)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /code

# Set environment variable to bypass model source connectivity check and cache models
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Limit internal threading so Paddle/OpenCV/BLAS don't spawn many threads
# on a low-memory/low-CPU instance (helps avoid OOM kills on free tiers)
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV FLAGS_use_mkldnn=false

# Copy requirements and install
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application files
COPY . /code

# Run uvicorn with a single worker (loads the OCR model only once)
# and a longer keep-alive/timeout so the first request after the
# free-tier instance wakes up from sleep doesn't get cut off.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "120"]
