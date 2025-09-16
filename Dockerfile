FROM python:3.11-bookworm

# Install system dependencies and build tools
# 'gcc' is the C compiler, 'python3-dev' includes Python header files
# 'libpq-dev' is required for psycopg2 to compile
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .