from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("C:\\Users\\Bianca\\PycharmProjects\\ACABI\\Face_Mask_Detection\\checkpoints\\test\\weights\\best.pt")
    results = model.val(data='C:\\Users\\Bianca\\PycharmProjects\\ACABI\\Face_Mask_Detection\\mask-dataset\\data.yaml', split='val')
