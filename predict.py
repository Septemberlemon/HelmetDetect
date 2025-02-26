from ultralytics import YOLO, settings


model = YOLO(settings["runs_dir"] + "/detect/yolo11s-helmet-detect/weights/best.pt")
model.to("cuda")

results = model.predict("sample1.png")
for result in results:
    result.show()
    # result.save("result.png")
