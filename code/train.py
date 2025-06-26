import os.path

import torch
import wandb
from ultralytics import YOLO
from utils import load_config

def train():
    config = load_config.load_config()

    # Initialize your Weights and Biases project
    wandb.init(
        entity=config["wandb"]["entity"],
        project=config["wandb"]["project"],  # Project name in wandb
        name=config["wandb"]["name"],  # Name of the run
        config={
            "epochs": 200,
            "batch_size": 10,
            "model": "last.pt",  # You can change this to other YOLOv8 variants
        }
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the pre-trained model
    model = YOLO(config["models"]["yolo8"])
    model.to(device)

    # Start training
    results = model.train(
        # Path to the dataset.yaml file
        data=os.path.join(config["dataset"], "data.yaml"),
        epochs=100,  # Number of epochs
        batch=10,  # Batch size
        save=True,  # Save checkpoints after each epoch
        device=device,
        # Directory where the model checkpoints will be saved
        project=config["checkpoints"],
        name=config["wandb"]["name"],  # Name of the training run folder
        optimizer='Adam',  # Optional: you can specify optimizer
        workers=2,  # Number of data loading workers
        verbose=True,  # Display training progress
        imgsz=800
    )

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    train()

