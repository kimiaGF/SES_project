# Use a lightweight Python image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the script and dependencies into the container
COPY ./script.py /app/script.py
COPY ./requirements.txt /app/requirements.txt
COPY ./cdd.txt /app/cdd.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the e