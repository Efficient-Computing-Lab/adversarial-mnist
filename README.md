# Enabling Adversarial Robustness in AI Models Through Kubeflow MLOps

This repository is presenting a way to add adversarial robustness in AI models by using the established MLOps tool named Kubeflow.
For the purposes of the experimentation this repository uses the MNIST dataset and a MNIST classifier model.


## Experiment Scenarios

1. **Train Model A** on clean MNIST 
2. **Attack Model A** with FGSM (accuracy drops)
3. **Generate PGD adversarial dataset** using frozen Model A or on the fly
4. **Retrain Model A′** on the PGD dataset (starting from A's weights)
5. **Evaluate Model A′** against FGSM (accuracy recovers)


## Architecture
This project builds upon the [repository](https://github.com/Efficient-Computing-Lab/kubeflow-mlops-architecture).
Follow the instructions of this project in order to install the Kubeflow setup.

The architecture uses **Kubeflow** as the primary component for training machine learning models. It also provides a **dataset registry** where users can store datasets intended for use during the training process.

Additionally, the setup integrates the **Kubernetes NVIDIA plugin** and **Kyverno** to enable temporary pods spawned by Kubeflow to access **GPU resources**.

### Normal Usage
The normal usage of this architecture expects the user to push the dataset on the registry, initiate the
training of an AI model through Kubeflow and then manualy
deploy a pod to perform the inference. All the trained models
along with their registered accuracies are stored in Kubernetes
volume.

```shell
cd trainer
python3 train.py
```
![Normal Usage](Kubeflow%20Adversarial%20Security-normal-usage.png)
### Attack Usage
The attack scenario for our case is an insider threat. This
means that a malicious user has somehow access to deploy a
pod and retrieve a trained model from the volume.
The purpose of the attack is to generate synthetic data similar
to the ones that the trained model understand and use them
to deteriorate the accuracy of the trained model.

To deploy the attack pod:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: attacker
  namespace: kubeflow
  labels:
    app: attacker
spec:
  restartPolicy: Never
  runtimeClassName: nvidia   # 👈 add this
  containers:
    - name: attacker
      image: gkorod/attacker:v1.0
      imagePullPolicy: Always
      resources:
        limits:
          nvidia.com/gpu: 1   # 👈 request GPU
      env:
        - name: SELECTOR
          value: "validate_defence"
        - name: DATA_URL
          value: "http://147.102.19.170:4422/download/classifier"
        - name: MODEL_PATH
          value: "/trained_models/classifier/model_a.pth"
        - name: MODEL_WITH_DEFENCE
          value: "/trained_models/classifier_retrained_adv/model_a.pth"
        - name: BATCH_SIZE
          value: "64"
        - name: EPSILON
          value: "0.15"
        - name: EVAL_SAMPLES
          value: "5000"
      volumeMounts:
        - name: trained-models-volume
          mountPath: /trained_models
        - mountPath: "/datasets"
          name: data-volume
  volumes:
    - name: trained-models-volume
      persistentVolumeClaim:
        claimName: trained-models
    - name: data-volume
      persistentVolumeClaim:
        claimName: datasets
```
![Attack Usage](Kubeflow%20Adversarial%20Security-attack-usage.png)
### Defence Usage
The defense  in our scenario being triggered in the inference phase.
The inference pod is able to retrieve the trained model
and the registered accuracy. This pod checks if the accuracy
is starting to drop and in that case is triggering a component
name defender. The defender component is able to start the
defence by using Kubeflow.

To deploy the defence service:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: defender
  namespace: kubeflow
  labels:
    app: defender
spec:
  replicas: 1
  selector:
    matchLabels:
      app: defender
  template:
    metadata:
      labels:
        app: defender
    spec:
      containers:
        - name: defender
          image: gkorod/defender:v1.0
          imagePullPolicy: Always
          ports:
            - containerPort: 5000
          env:
            - name: DATA_URL
              value: "http://147.102.19.170:4422/download/"
            - name: INITIALIZER_URL
              value: "http://147.102.19.170:5005/submit"
          volumeMounts:
            - name: trained-models-volume
              mountPath: /trained_models

      volumes:
        - name: trained-models-volume
          persistentVolumeClaim:
            claimName: trained-models
```
![Defence Usage](Kubeflow%20Adversarial%20Security-defence-usage.png)