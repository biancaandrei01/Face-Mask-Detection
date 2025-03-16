import torch
import wandb
from ultralytics import YOLO

if __name__ == "__main__":
    # Initialize your Weights and Biases project
    wandb.init(
        entity="transformers_3",
        project="Face Mask Detection",  # Project name in wandb
        name="test",  # Name of the run
        config={
            "epochs": 200,
            "batch_size": 10,
            "model": "last.pt",  # You can change this to other YOLOv8 variants
        }
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load the pre-trained model
    model = YOLO("yolov8n.pt")
    model.to(device)

    # Start training
    results = model.train(
        # Path to the dataset.yaml file
        data='C:\\Users\\Bianca\\PycharmProjects\\ACABI\\Face_Mask_Detection\\mask-dataset\\data.yaml',
        epochs=100,  # Number of epochs
        batch=10,  # Batch size
        save=True,  # Save checkpoints after each epoch
        device=device,
        # Directory where the model checkpoints will be saved
        project="checkpoints",
        name="test",  # Name of the training run folder
        optimizer='Adam',  # Optional: you can specify optimizer
        workers=2,  # Number of data loading workers
        verbose=True,  # Display training progress
        imgsz=800
    )

    # Finish the wandb run
    wandb.finish()

