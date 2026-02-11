from numpy.ma.core import outer
from setuptools.archive_util import unpack_zipfile

from fgsm import FGSM
from classifier import MNISTClassifier
from evaluation import evaluate_attack
from visualization import visualize_adversarial_examples
import os
import torch
from standard_training import get_mnist_loaders
import requests
import zipfile




print("\n" + "=" * 60)
print("Attack Model A with FGSM")
print("=" * 60)

model_A = MNISTClassifier()
model_path = os.getenv("MODEL_PATH","EMPTY")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = int(os.getenv("BATCH_SIZE",64))
epsilon = float(os.getenv("EPSILON",0.15))
eval_samples = int(os.getenv("EVAL_SAMPLES",5000))
url_dataset = os.getenv("DATA_URL","EMPTY")

def unzip(output_file):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)  # create folder if not exists

    # Extract ZIP
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"Extracted contents to '{output_dir}'")

    # List all extracted files
    for root, dirs, files in os.walk(output_dir):
        for name in files:
            file_path = os.path.join(root, name)
            print(f"- {file_path}")

    # Find the first child folder inside output_dir
    child_folders = [f for f in os.listdir(output_dir)
                     if os.path.isdir(os.path.join(output_dir, f))]

    if child_folders:
        # Return the first child folder path
        child_path = os.path.join(output_dir, child_folders[0])
        print(f"Found child folder: {child_path}")
        return child_path
    else:
        # No child folder found, return output_dir itself
        return output_dir

def download_dataset():
    output_file = "data.zip"

    with requests.get(url_dataset, stream=True) as r:
        r.raise_for_status()
        with open(output_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("Downloaded:", output_file)
    return output_file

print(f"Using device: {device}")
output_file = download_dataset()
unzip_folder = unzip(output_file)

train_loader, test_loader = get_mnist_loaders(
        batch_size=batch_size,
        data_dir= unzip_folder
    )


if model_path != "EMPTY":
    print(f"Loading pre-trained Model A from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model_A.load_state_dict(checkpoint['model_state_dict'])
    model_A.to(device)
# Create FGSM attack using Model A
generated_data = FGSM(model_A, epsilon= epsilon)

# Evaluate FGSM attack on Model A
results_A = evaluate_attack(
    model_A, generated_data, test_loader, device,
    num_samples=eval_samples
)

print(f"Model A under FGSM (epsilon={epsilon}):")
print(f"  Clean accuracy: {results_A['clean_accuracy']:.2f}%")
print(f"  FGSM accuracy:  {results_A['adversarial_accuracy']:.2f}%")
print(f"  Attack success: {results_A['attack_success_rate']:.2f}%")