from ultralytics import YOLO, settings


if __name__ == '__main__':
    model = YOLO(settings["runs_dir"] + "/detect/yolov8n-helmet-detect/weights/best.pt")
    model.val(data="dataset.yaml")
