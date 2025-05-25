# Multi-stage build for production optimization
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-mock pytest-cov

# Production stage
FROM base as production

# Copy application code
COPY app ./app
COPY ontology_config.yaml .
COPY go.json .

# Create data directory for ontology storage
RUN mkdir -p /app/data

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create empty version files with proper permissions
RUN touch /app/ontology_versions.json && \
    touch /app/embeddings_config.yaml || true

# Set ownership and permissions
RUN chown -R appuser:appuser /app && \
    chmod -R 755 /app && \
    chmod 664 /app/ontology_versions.json && \
    chmod 664 /app/embeddings_config.yaml || true

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Default to production
FROM production
