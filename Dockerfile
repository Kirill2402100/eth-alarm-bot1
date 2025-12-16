FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# На реальном рантайме /data перекрывается volume, но mkdir не мешает
RUN mkdir -p /data /data/out

CMD ["python", "-u", "-m", "scripts.run_grid", "--config", "configs/base.yaml"]
