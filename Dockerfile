# ==============================================================================
# Stage 1: Build stage (Installs dependencies into a isolated virtualenv)
# ==============================================================================
FROM python:3.11-slim AS builder

# Prevent Python from writing .pyc files & buffer output, disable pip cache
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ==============================================================================
# Stage 2: Runtime stage (Minimal image running non-root)
# ==============================================================================
FROM python:3.11-slim AS runtime

# Set runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

WORKDIR /app

# Install git — required by gitpython at runtime (it wraps the git CLI).
# Acquire::ForceIPv4=true prevents apt from stalling on Docker Desktop /
# Windows where IPv6 routing to Debian mirrors is broken.
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4 && \
    apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Create non-root system user and group for security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -d /app appuser

# Copy installed virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code
COPY app ./app
COPY parser ./parser
COPY worker ./worker

# Pre-create the repos temp dir so appuser can write to it without
# needing root at runtime
RUN mkdir -p /tmp/repos && chown appuser:appgroup /tmp/repos

# Set permissions for non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose Uvicorn default port
EXPOSE 8000

# Entrypoint command to start Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
