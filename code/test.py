from ultralytics import YOLO
import os.path
from utils import load_config

def test():
    config = load_config.load_config()

    model = YOLO(os.path.join(config["checkpoints"], config["wandb"]["name"], "weights", "best.pt"))
    results = model.val(data=os.path.join(config["dataset"], "data.yaml"),
                        split='val')

if __name__ == "__main__":
    test()
