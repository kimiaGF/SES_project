# Use a lightweight Python image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the script and dependencies into the container
COPY . /app/.

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point for the container to execute the script
ENTRYPOINT ["python", "/app/script.py"]

