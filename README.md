# Adversarial Robustness Pipeline for MNIST


## What This Project Does

1. **Train Model A** on clean MNIST 
2. **Attack Model A** with FGSM (accuracy drops)
3. **Generate PGD adversarial dataset** using frozen Model A or on the fly
4. **Retrain Model A′** on the PGD dataset (starting from A's weights)
5. **Evaluate Model A′** against FGSM (accuracy recovers)


