import os
import torch
import shutil
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import struct
import gzip
import numpy as np

from fgsm import FGSM
from classifier import MNISTClassifier
from evaluation import evaluate_attack
from standard_training import get_mnist_loaders

# ----------------------------
# Config / Env Variables
# ----------------------------
model_path = os.getenv("MODEL_PATH", "EMPTY")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = int(os.getenv("BATCH_SIZE", 64))
epsilon = float(os.getenv("EPSILON", 0.15))
eval_samples = int(os.getenv("EVAL_SAMPLES", 5000))
url_dataset = os.getenv("DATA_URL", "EMPTY")

# ----------------------------
# Helper Functions
# ----------------------------
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

def zip_output(zip_path="data.zip"):
    output_dir = Path("output").resolve()
    if not output_dir.exists() or not output_dir.is_dir():
        raise RuntimeError("output directory not found")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(output_dir)
                zipf.write(file_path, arcname)

    print(f"Zipped contents of '{output_dir}' to '{zip_path}'")
    for name in zipfile.ZipFile(zip_path).namelist():
        print(f"- {name}")

    return zip_path

def save_malicius_dataset(result):
    """
    Save FGSM dataset as MNIST-style ubyte.gz files compatible with torchvision.datasets.MNIST.
    """
    output_dir = Path("output").resolve()

    # Clear output contents
    if output_dir.exists() and output_dir.is_dir() and output_dir.name == "output":
        for item in output_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            else:
                shutil.rmtree(item)
    else:
        raise RuntimeError("Output directory not found or unsafe path")

    # Create output/data folder
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    images = result['adv_images'].numpy()  # [N, 1, 28, 28]
    labels = result['labels'].numpy()

    # ------------------------------
    # Save images as ubyte.gz
    # ------------------------------
    images_file = data_dir / "train-images-idx3-ubyte.gz"
    with gzip.open(images_file, 'wb') as f:
        f.write(struct.pack(">I", 2051))            # magic number
        f.write(struct.pack(">I", images.shape[0])) # number of images
        f.write(struct.pack(">I", images.shape[2])) # rows
        f.write(struct.pack(">I", images.shape[3])) # cols
        for i in range(images.shape[0]):
            img_bytes = (images[i, 0] * 255).astype(np.uint8)
            f.write(img_bytes.tobytes())

    # ------------------------------
    # Save labels as ubyte.gz
    # ------------------------------
    labels_file = data_dir / "train-labels-idx1-ubyte.gz"
    with gzip.open(labels_file, 'wb') as f:
        f.write(struct.pack(">I", 2049))            # magic number
        f.write(struct.pack(">I", labels.shape[0])) # number of labels
        f.write(labels.astype(np.uint8).tobytes())

    print(f"Saved {images.shape[0]} adversarial images and labels to '{data_dir}'")

    # Zip the folder
    zip_path = zip_output("data.zip")
    print(f"Dataset zipped at: {zip_path}")

# ----------------------------
# Main Script
# ----------------------------
print("\n" + "=" * 60)
print("Attack Model A with FGSM")
print("=" * 60)
print(f"Using device: {device}")

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

# 4️⃣ Create FGSM attack object
attack = FGSM(model_A, epsilon=epsilon)

# 5️⃣ Generate adversarial dataset
print("Generating FGSM adversarial dataset...")
all_clean_images = []
all_adv_images = []
all_labels = []

for images, labels in tqdm(test_loader, desc="Generating FGSM samples"):
    images, labels = images.to(device), labels.to(device)
    adv_images = attack.attack(images, labels)

    all_clean_images.append(images.cpu())
    all_adv_images.append(adv_images.cpu())
    all_labels.append(labels.cpu())

generated_data = {
    'clean_images': torch.cat(all_clean_images, dim=0),
    'adv_images': torch.cat(all_adv_images, dim=0),
    'labels': torch.cat(all_labels, dim=0),
    'epsilon': epsilon,
    'attack': 'FGSM'
}

# 6️⃣ Evaluate attack on Model A using the FGSM object
results_A = evaluate_attack(
    model_A, attack, test_loader, device,
    num_samples=eval_samples
)

print(f"Model A under FGSM (epsilon={epsilon}):")
print(f"  Clean accuracy: {results_A['clean_accuracy']:.2f}%")
print(f"  FGSM accuracy:  {results_A['adversarial_accuracy']:.2f}%")
print(f"  Attack success: {results_A['attack_success_rate']:.2f}%")

# 7️⃣ Save & zip FGSM dataset in ubyte.gz format
save_malicius_dataset(generated_data)
