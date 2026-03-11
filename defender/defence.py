import json
import requests
import tarfile
import os


from kfp import dsl, compiler
from kfp.dsl import component, pipeline
from kfp import kubernetes
import kfp


pipeline_version = 1
# -------------------------------
# Component 1 — Download & unzip dataset
# -------------------------------


@component(
    packages_to_install=["requests"],
    base_image="python:3.12",
)
def download_dataset(output_dir: dsl.Output[dsl.Artifact], data_url: str):
    import requests
    import zipfile
    import os

    zip_path = "data.zip"

    # Download dataset
    with requests.get(data_url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

    print("Downloaded dataset")

    extract_root = output_dir.path
    os.makedirs(extract_root, exist_ok=True)

    # Unzip dataset
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_root)

    print("Extracted to:", extract_root)

    # Detect inner dataset folder
    child_folders = [
        f for f in os.listdir(extract_root)
        if os.path.isdir(os.path.join(extract_root, f))
    ]

    if child_folders:
        dataset_root = os.path.join(extract_root, child_folders[0])
        print("Detected dataset root:", dataset_root)
    else:
        dataset_root = extract_root
        print("Using root dataset:", dataset_root)

    print("Final dataset path used by pipeline:", dataset_root)


# -------------------------------
# Component 2 — Adversarial Retraining
# -------------------------------

@component(
    base_image="gkorod/defender:v1.0",
)
def adversarial_retraining(
    dataset_dir: dsl.Input[dsl.Artifact],
    batch_size: int,
    epsilon: float,
    epochs: int,
    learning_rate: float,
    alpha: float,
    pgd_steps: int,
    model_path: str,
    model_name: str
):
    import torch
    from classifier import MNISTClassifier
    from adversarial_retraining import retrain_on_the_fly
    from standard_training import get_mnist_loaders
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = dataset_dir.path

    print("Dataset directory received:", data_root)
    for file in os.listdir(model_path):
        if file.endswith(".pth"):
            found_model = os.path.join(model_path, file)
    print("Model path:", found_model)

    train_loader, test_loader = get_mnist_loaders(
        batch_size=batch_size,
        data_dir=data_root
    )

    model_A = MNISTClassifier()
    save_path = f"/trained_models/{model_name}_retrained_adv/model_a.pth"
    checkpoint = torch.load(found_model, map_location=device)
    model_A.load_state_dict(checkpoint["model_state_dict"])
    model_A.to(device)

    print("=" * 60)
    print("On-the-fly adversarial training")
    print("=" * 60)

    retrain_on_the_fly(
        model=model_A,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        lr=learning_rate,
        epsilon=epsilon,
        alpha=alpha,
        num_steps=pgd_steps,
        save_path=save_path
    )

    print("Retrained model saved to:", model_path)


# -------------------------------
# Pipeline
# -------------------------------

@pipeline
def adversarial_training_pipeline(
    data_url: str,
    batch_size: int,
    epsilon: float,
    epochs: int,
    learning_rate: float,
    alpha: float,
    pgd_steps: int,
    model_path: str,
    model_name: str,
):
    download_task = download_dataset(data_url=data_url)

    retrain_task = adversarial_retraining(
        dataset_dir=download_task.outputs["output_dir"],
        batch_size=batch_size,
        epsilon=epsilon,
        epochs=epochs,
        learning_rate=learning_rate,
        alpha=alpha,
        pgd_steps=pgd_steps,
        model_path=model_path,
        model_name=model_name
    )
    retrain_task.set_accelerator_type('nvidia.com/gpu')
    retrain_task.set_accelerator_limit(1)
    kubernetes.mount_pvc(
        retrain_task,
        pvc_name="trained-models",
        mount_path="/trained_models",
    )


# -------------------------------
# Compile pipeline
# -------------------------------

def initialize_defence_pipeline(model_name):
    global pipeline_version

    url_dataset = os.getenv("DATA_URL", "http://147.102.19.170:4422/download/")
    url_initializer = os.getenv("INITIALIZER_URL", "http://147.102.19.170:5005/submit")
    data_url = url_dataset + model_name
    model_path = f"/trained_models/{model_name}/"
    compiler.Compiler().compile(
        pipeline_func=adversarial_training_pipeline,
        package_path="pipeline.yaml",
        pipeline_parameters={
            "data_url": data_url,
            "batch_size": 64,
            "epsilon": 0.15,
            "epochs": 20,
            "learning_rate": 0.0001,
            "alpha": 0.1,
            "pgd_steps": 20,
            "model_path": model_path,
            "model_name": model_name
        }
    )

    with tarfile.open("pipeline.tar.gz", "w:gz") as tar:
        tar.add("pipeline.yaml", arcname="pipeline.yaml")

    json_info = {
        "user_namespace": "test",
        "experiment": "experiment_defence_test",
        "pipeline_name": "defence_pipeline",
        "job_name": "defence_job",
        "pipeline_version": str(pipeline_version)
    }

    with open("pipeline.tar.gz", "rb") as f:
        response = requests.post(
            url_initializer,
            files={"file": ("pipeline.tar.gz", f, "application/x-tar")},
            data={"json_data": json.dumps(json_info)}
        )

    print(response.text)

    # increment AFTER successful submission
    pipeline_version += 1