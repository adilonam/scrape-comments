FROM python:3.12-slim

# Install minimal required dependencies for Chrome and Tor
RUN apt-get update && apt-get install -y \
    tor \
    wget \
    gnupg2 \
    apt-transport-https \
    ca-certificates \
    && wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get install -y ./google-chrome-stable_current_amd64.deb \
    && rm google-chrome-stable_current_amd64.deb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Start Tor service and run the application
CMD ["sh", "-c", "service tor start && python src/app.py"]
