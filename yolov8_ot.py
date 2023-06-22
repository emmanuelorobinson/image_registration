from ultralytics import YOLO
import supervision as sv
import cv2
model = YOLO('./Weights/best.pt')

# # results = model.track('./DJI Dataset/DJI1_W.MP4', show=True)
# results = model.track('./output.mp4', show=True)
# print(results)

results = model('./Capture/0_frame0.jpg')
frame ='./Capture/0_frame0.jpg'

box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

result = model(frame, agnostic_nms=True)[0]
detections = sv.Detections.from_yolov8(result)
labels = [
  f"{model.model.names[class_id]} {confidence:0.2f}"
  for xyxy, masks, confidence, class_id, tracker_id in detections
  ]
frame = box_annotator.annotate(
  scene=frame, 
  detections=detections, 
  labels=labels
) 

cv2.imshow("yolov8", frame)