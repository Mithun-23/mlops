FROM python:3.13-slim

WORKDIR /app

# Install system dependencies (especially libgomp1 for ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code and model files
COPY . .

# Railway automatically handles port binding through the PORT environment variable.
ENV PORT=8000
EXPOSE $PORT

# Run the Streamlit server
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0
