# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install Poetry
COPY install-poetry.py ./
RUN python install-poetry.py --yes && \
    export PATH="/root/.local/bin:$PATH" && \
    poetry config virtualenvs.create false

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY SafeCast ./SafeCast
COPY backend ./backend
COPY assets ./assets
COPY yolo11n*.pt ./
COPY README.md LICENSE.txt ./

# Install only runtime dependencies
RUN export PATH="/root/.local/bin:$PATH" && \
    poetry install --only main

# Expose the port Flask runs on
EXPOSE 5000

# Run the app
CMD ["python3", "SafeCast/app.py"]
