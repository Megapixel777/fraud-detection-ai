FROM python:3.12-slim-bookworm

# Install Java 17 required by PySpark
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

# Install API dependencies
COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application and model
COPY src/ ./src/
COPY models/ ./models/

# API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]