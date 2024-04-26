# Use an official Python runtime as a parent image
#FROM python:3.11.5-slim
#FROM ultralytics/ultralytics
#FROM nvidia/cuda:12.4.1-base-ubi8

#FROM nvcr.io/nvidia/pytorch:24.03-py3
FROM nvidia/cuda:12.4.1-base-ubuntu22.04
# Set the working directory in the container
WORKDIR /app


# Copy the requirements file into the container
COPY requirements.txt .


######################
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6\
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*



########

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

#RUN pip3 install -r requirements.txt

#RUN pip3 install flask
# Copy the inference script into the container
COPY inference.py .
#COPY best.pt .

EXPOSE 8080

ENTRYPOINT ["python3", "inference.py"]
