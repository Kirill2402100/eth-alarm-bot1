FROM python:3.12-slim

WORKDIR /app

# Системные зависимости (если надо для pandas/parquet/сборки колес)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

# Код
COPY . .

# Важно: /data создаём НА СТАРТЕ, чтобы это было внутри смонтированного volume
CMD ["sh", "-lc", "mkdir -p /data /data/out && exec python -u -m scripts.run_grid --config configs/base.yaml"]
