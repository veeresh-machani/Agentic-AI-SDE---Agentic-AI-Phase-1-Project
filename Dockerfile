FROM python:3.11-slim

# Create a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies first for Docker layer caching
COPY --chown=user backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user . /app

# Hugging Face Spaces default port
EXPOSE 7860

# Run uvicorn on port 7860 (or $PORT if set by another platform)
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
