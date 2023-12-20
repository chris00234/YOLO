from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictior
import cv2

model = YOLO("yolov8s.pt")
results = model.predict(source = "0", show = True)
print(results)