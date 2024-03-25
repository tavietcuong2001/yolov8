from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
import os

model = YOLO("yolov8n.pt")
names = model.names
crop_dir_name = "output/object_cropping"
if not os.path.exists(crop_dir_name):
    os.mkdir(crop_dir_name)

idx = 0
im0 = cv2.imread("data/image.jpg")
results = model.predict(im0)
boxes = results[0].boxes.xyxy.cpu().tolist()
clss = results[0].boxes.cls.cpu().tolist()
annotator = Annotator(im0, line_width=2, example=names)

if boxes is not None:
    for box, cls in zip(boxes, clss):
        idx += 1
        annotator.box_label(box, label=names[int(cls)])

        crop_obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        cv2.imwrite(os.path.join(crop_dir_name, str(idx)+".png"), crop_obj)
