FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Postgres (Debian default) + build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    postgresql postgresql-client postgresql-contrib \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app

# Copy aisuite config to home directory
RUN mkdir -p /root && cp /app/.aisuite.yaml /root/.aisuite.yaml || true

# Your entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 5432
CMD ["/entrypoint.sh"]
