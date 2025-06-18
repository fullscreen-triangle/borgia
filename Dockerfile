# Multi-stage build for Borgia
FROM rust:1.70-bullseye as rust-builder

# Install system dependencies for cheminformatics
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Rust project files
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build Rust application
RUN cargo build --release

# Python stage
FROM python:3.11-bullseye as python-builder

# Install system dependencies for scientific computing and cheminformatics
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libpq-dev \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python project files
COPY pyproject.toml requirements.txt ./
COPY python ./python

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Final runtime stage
FROM python:3.11-slim-bullseye

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash borgia

# Set working directory
WORKDIR /app

# Copy built Rust binary
COPY --from=rust-builder /app/target/release/borgia /usr/local/bin/borgia

# Copy Python installation
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy application files
COPY . .

# Change ownership to borgia user
RUN chown -R borgia:borgia /app

# Switch to non-root user
USER borgia

# Create necessary directories
RUN mkdir -p logs data models cache results

# Set environment variables
ENV PYTHONPATH=/app/python
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "borgia.api"] 