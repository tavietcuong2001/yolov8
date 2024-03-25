from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
im0 = cv2.imread("data/image.jpg")
results = model.predict(im0)
boxes = results[0].boxes.xyxy.cpu().tolist()
clss = results[0].boxes.cls.cpu().tolist()

if boxes is not None:
    for box, cls in zip(boxes, clss):
        if cls==0:   # blurring person object
            obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            blur_obj = cv2.blur(obj, (50, 50))

            im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = blur_obj

cv2.imwrite("output/object_blurring.png", im0)