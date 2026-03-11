import requests
import zipfile
import os
import torch
from standard_training import get_mnist_loaders
from classifier import MNISTClassifier
import json
import sys

model_path = os.getenv("MODEL_PATH", "EMPTY")
clean_accuracy = os.getenv("CLEAN_ACCURACY", "EMPTY")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = int(os.getenv("BATCH_SIZE", 64))
epsilon = float(os.getenv("EPSILON", 0.15))
eval_samples = int(os.getenv("EVAL_SAMPLES", 5000))
url_dataset = os.getenv("DATA_URL", "EMPTY")
defender_url = os.getenv("DEFENDER_URL", "EMPTY")

def evaluate(model, test_loader, device,clean_accuracy_value):
    """Evaluate model on clean test data."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    current_accuracy = correct / total
    if current_accuracy < clean_accuracy_value:
        print(f"model has been attacked")
        payload = {
            "model_name": "classifier",
            "dataset": "mnist"
        }
        response = requests.post(defender_url, json=payload)
        print("Status:", response.status_code)
        print("Response:", response.text)
        sys.exit(0)
    return current_accuracy

def download_dataset():
    output_file = "data.zip"
    with requests.get(url_dataset, stream=True) as r:
        r.raise_for_status()
        with open(output_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded: {output_file}")
    return output_file

def unzip(output_file):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(output_file, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"Extracted contents to '{output_dir}'")
    for root, dirs, files in os.walk(output_dir):
        for name in files:
            print(f"- {os.path.join(root, name)}")

    child_folders = [f for f in os.listdir(output_dir)
                     if os.path.isdir(os.path.join(output_dir, f))]
    if child_folders:
        child_path = os.path.join(output_dir, child_folders[0])
        print(f"Found child folder: {child_path}")
        return child_path
    return output_dir

# 1️⃣ Download and unzip dataset
output_file = download_dataset()
unzip_folder = unzip(output_file)

# 2️⃣ Load MNIST data loaders
train_loader, test_loader = get_mnist_loaders(
    batch_size=batch_size,
    data_dir=unzip_folder
)

# 3️⃣ Load pre-trained Model A
model_A = MNISTClassifier()
if model_path != "EMPTY":
    print(f"Loading pre-trained Model A from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model_A.load_state_dict(checkpoint['model_state_dict'])
model_A.to(device)
model_A.eval()
with open(clean_accuracy, "r") as f:
    recorded_accuracy = json.load(f)
clean_accuracy_value =recorded_accuracy.get("accuracy")
evaluate(model_A, test_loader, device, clean_accuracy_value)