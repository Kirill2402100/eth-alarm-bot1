FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости (если надо для pandas/parquet)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Создаём папку под volume (иначе Railway пишет "missing dir")
RUN mkdir -p /data /data/out

# Запускаем напрямую
CMD ["python", "-u", "-m", "scripts.run_grid", "--config", "configs/base.yaml"]
