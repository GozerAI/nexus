FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY src/ ./src/
COPY config/ ./config/

# Set Python path
ENV PYTHONPATH=/app/src

# Run the ensemble core module
CMD ["python", "-m", "nexus.ensemble_core"]
