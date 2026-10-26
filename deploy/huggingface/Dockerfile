# The public demo on Hugging Face Spaces (free CPU): Forma from GitHub at a pinned commit, rules-only, no LLM.
# Single-line commands and no brackets, so the file can be pasted into the Space's web editor as is.
FROM python:3.12-slim
ARG FORMA_COMMIT=main
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/Manvith-d/forma /app && git -C /app checkout $FORMA_COMMIT
WORKDIR /app
ENV PYTHONUNBUFFERED=1 FORMA_HOME=/tmp/forma FORMA_MODEL_CACHE=/opt/models FORMA_LLM=none FORMA_DEMO=1
RUN pip install --no-cache-dir -e . && forma pipeline
RUN useradd -m -u 1000 user
USER user
EXPOSE 7860
CMD forma serve --host 0.0.0.0 --port 7860
