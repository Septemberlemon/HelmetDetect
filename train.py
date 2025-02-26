from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("yolo11s.pt")
    model.to("cuda")
    model.train(data="dataset.yaml", epochs=100)
