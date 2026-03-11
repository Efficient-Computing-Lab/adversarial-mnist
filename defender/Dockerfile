FROM python:3.12

# Set working directory
WORKDIR /app


# Copy requirements file
COPY adversarial_retraining.py .
COPY classifier.py .
COPY api.py .
COPY defence.py .
COPY requirements.txt .
COPY standard_training.py .
COPY pgd.py .

# Install Python dependencies
RUN pip install -r requirements.txt

# Optional: set default command
CMD ["python", "api.py"]