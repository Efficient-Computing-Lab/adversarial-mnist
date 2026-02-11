FROM python:3.12

# Set working directory
WORKDIR /app


# Copy requirements file
COPY classifier.py .
COPY standard_training.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: set default command
# CMD ["python", "fgsm-attack.py"]