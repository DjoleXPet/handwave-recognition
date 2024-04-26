FROM nvidia/cuda:12.4.1-base-ubuntu22.04

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6\
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r requirements.txt

COPY inference.py .

EXPOSE 8080

ENTRYPOINT ["python3", "inference.py"]
