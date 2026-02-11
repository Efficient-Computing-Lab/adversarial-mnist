FROM python:3.12

# Set working directory
WORKDIR /app


# Copy requirements file
COPY classifier.py .
COPY evaluation.py .
COPY fgsm.py .
COPY fgsm-attack.py .
COPY visualization.py .
COPY requirements.txt .
COPY standard_training.py .


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: set default command
CMD ["python", "fgsm-attack.py"]