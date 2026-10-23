FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    FORMA_HOME=/app/runtime \
    FORMA_MODEL_CACHE=/opt/models
WORKDIR /app
COPY pyproject.toml README.md LICENSE sources.yaml ./
COPY knowledge knowledge
COPY src src
RUN pip install --no-cache-dir -e .
# Download the public sources and build the warehouse into the image: bronze -> silver -> gold -> 13 checks
RUN forma pipeline
COPY web web
RUN useradd --create-home app && mkdir -p runtime && chown app runtime
USER app

EXPOSE 8000
CMD ["forma", "serve", "--host", "0.0.0.0", "--port", "8000"]
