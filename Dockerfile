FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7865 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-venv \
        python3.13-dev \
        python3.13-distutils \
        build-essential \
        git \
        curl \
        wget && \
    python3.13 -m ensurepip --upgrade && \
    python3.13 -m pip install --no-cache-dir --upgrade pip && \
    ln -sf /usr/bin/python3.13 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.13 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user to run the application
RUN useradd -m -u 1001 appuser

# Set working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/ace-step/ACE-Step.git .

# Install specific PyTorch version compatible with CUDA 12.6
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir hf_transfer peft && \
    python -m pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
RUN python -m pip install --no-cache-dir .

# Ensure target directories for volumes exist and have correct initial ownership
RUN mkdir -p /app/outputs /app/checkpoints /app/logs && \
    chown -R appuser:appuser /app/outputs /app/checkpoints /app/logs

# Change ownership of app files to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 7865

VOLUME [ "/app/checkpoints", "/app/outputs", "/app/logs" ]

# Set healthcheck
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=5 \
  CMD curl -f http://localhost:7865/ || exit 1

# Command to run the application with GPU support
CMD ["python3", "acestep/gui.py", "--server_name", "0.0.0.0", "--bf16", "true"]
