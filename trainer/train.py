import json
import requests
import tarfile
import os
from kfp import dsl, compiler
from kfp.dsl import component, pipeline
from kfp import kubernetes
import kfp


# -------------------------------
# Component 1 — Download & unzip dataset
# -------------------------------

@component(
    packages_to_install=["requests"],
    base_image="python:3.12",
)
def download_dataset(output_dir: dsl.Output[dsl.Artifact], data_url: str):
    import requests, zipfile, os

    zip_path = "data.zip"

    with requests.get(data_url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

    os.makedirs(output_dir.path, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir.path)

    print("Extracted to:", output_dir.path)


# -------------------------------
# Component 2 — Train Model A (FIXED)
# -------------------------------

@component(
    base_image="gkorod/trainer:v1.0",
)
def train_model_a(
    dataset_dir: dsl.Input[dsl.Artifact],
    epochs: int,
    batch_size: int,
    lr: float,
):
    import os
    import torch
    from classifier import MNISTClassifier
    from standard_training import get_mnist_loaders, train_standard

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = dataset_dir.path

    train_loader, test_loader = get_mnist_loaders(
        batch_size=batch_size,
        data_dir=data_root
    )

    model = MNISTClassifier().to(device)

    # ✅ WRITE DIRECTLY TO PVC
    os.makedirs("/trained_models", exist_ok=True)
    save_path = "/trained_models/model_a.pth"

    train_standard(
        model,
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        lr=lr,
        save_path=save_path
    )

    print("Model saved to PVC:", save_path)


# -------------------------------
# Pipeline
# -------------------------------

@pipeline
def fgsm_training_pipeline(
    data_url: str = 'http://147.102.19.170:4422/download/classifier',
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
):
    download_task = download_dataset(data_url=data_url)

    train_task = train_model_a(
        dataset_dir=download_task.outputs["output_dir"],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # ✅ Persist trained model
    kubernetes.mount_pvc(
        train_task,
        pvc_name="trained-models",
        mount_path="/trained_models",
    )


# -------------------------------
# Compile pipeline
# -------------------------------

compiler.Compiler().compile(
    fgsm_training_pipeline,
    package_path="pipeline.yaml"
)


# -------------------------------
# Package pipeline
# -------------------------------

with tarfile.open("pipeline.tar.gz", "w:gz") as tar:
    tar.add("pipeline.yaml", arcname="pipeline.yaml")


# -------------------------------
# Upload & run
# -------------------------------

json_info = {
    "user_namespace": "test",
    "experiment": "experiment_test",
    "pipeline_name": "fgsm_mnist_pipeline",
    "job_name": "fgsm_training_job",
    "pipeline_version": "1"
}

files = {
    "file": ("pipeline.tar.gz", open("pipeline.tar.gz", "rb"), "application/x-tar")
}

response = requests.post(
    "http://147.102.19.170:5005/submit",
    files=files,
    data={"json_data": json.dumps(json_info)}
)

print(response.text)
