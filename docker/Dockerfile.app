FROM python:3.10-slim

# Avoid Python writing .pyc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps if needed (uncomment if you hit missing packages)
# RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY docker/requirements.app.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.app.txt

# Copy your source code
COPY app app
COPY utils utils

# Expose FastAPI port
EXPOSE 8001

# Default entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
