# CUDA 13.0 userspace inside container
FROM nvidia/cuda:13.0.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev gcc g++ make libpq-dev git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# PyTorch built for CUDA 13.0
RUN python3 -m pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install PyTorch Geometric and dependencies
RUN pip install torch_geometric

#RUN pip install torch-scatter

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

