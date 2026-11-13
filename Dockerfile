# Dockerfile for DS108 Rainfall Forecasting Serving Engine
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install serving dependencies
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Copy source code and model artifacts
COPY src/ /app/src/
COPY artifacts/models/ /app/artifacts/models/
COPY reports/test_benchmark/ /app/reports/test_benchmark/

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI app with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

