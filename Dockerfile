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

# Prevent Paddle's CPU allocator from growing its internal memory pool
# aggressively/without limit. This keeps memory usage tighter and more
# predictable across many sequential requests on a 512MB instance.
ENV FLAGS_allocator_strategy=naive_best_fit
ENV FLAGS_fraction_of_cpu_memory_to_use=0.2

# Copy requirements and install
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application files
COPY . /code

# Run with gunicorn managing a single uvicorn worker.
# --max-requests recycles (fully restarts) the worker process after it
# has handled a number of requests, which releases any memory that
# PaddlePaddle's C++ allocator accumulated/fragmented and didn't give
# back to the OS. This prevents the slow memory build-up that leads to
# an OOM crash (and the 502s) after a bunch of scans.
CMD ["gunicorn", "app:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "1", \
     "--bind", "0.0.0.0:7860", \
     "--timeout", "120", \
     "--max-requests", "30", \
     "--max-requests-jitter", "5"]
