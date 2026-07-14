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

# Copy requirements and install
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application files
COPY . /code

# Run uvicorn on port 7860 (Hugging Face expects port 7860)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
