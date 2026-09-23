FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for LightGBM, XGBoost, and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Cloud Run injects $PORT (defaults to 8080), local docker uses 8000
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT}
