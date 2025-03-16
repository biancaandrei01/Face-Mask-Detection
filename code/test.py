from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("checkpoints\\???\\weights\\best.pt")

    results = model.val(data='???\\data.yaml', split='val')
