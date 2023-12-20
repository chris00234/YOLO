from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# model = YOLO("yolov8s.pt")
# model.train(data = "dataset.yaml", epochs= 3)

cap = cv2.VideoCapture(0)
# H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'runs', 'detect', 'train10', 'weights', 'best.pt')

# # Load a model
model = YOLO(model_path)  # load a custom model



threshold = 0.0

while True:
    ret, frame = cap.read()
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    # out.write(frame)
    cv2.imshow("YOLO V8", frame)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()