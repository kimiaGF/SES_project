# Use a lightweight Linux base image with Python installed
FROM debian:bullseye-slim

# Install Python and necessary tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the script and dependencies into