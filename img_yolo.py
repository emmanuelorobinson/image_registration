import uuid
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
WIDTH = 1080
HEIGHT = 1350


def main():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('result.mp4', fourcc, 30, (WIDTH, HEIGHT))
    
    # define resolution
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('./output.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # specify the model
    model = YOLO("./Weights/best.pt")

    # customize the bounding box
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )


    # while True:
    #     ret, frame = cap.read()
    #     result = model(frame, agnostic_nms=True)[0]
    #     detections = sv.Detections.from_yolov8(result)
    #     labels = [
    #         f"{model.model.names[class_id]} {confidence:0.2f} {tracker_id}"
    #         for xyxy, masks, confidence, class_id, tracker_id in detections
    #     ]
    #     frame = box_annotator.annotate(
    #         scene=frame, 
    #         detections=detections, 
    #         labels=labels
    #     ) 
        
    #     video.write(frame)
        
    #     cv2.imshow("yolov8", frame)

    #     if (cv2.waitKey(30) == 27): # break with escape key
    #         break

    # initialize an empty dictionary to store trackers
    trackers = {}

    while True:
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = []
        for xyxy, masks, confidence, class_id, _ in detections:
            # check if we already have a tracker for this detection
            if class_id in trackers:
                # update the tracker with the new frame
                # get the bounding box coordinates from the tracker
                # bbox = trackers[class_id].get_position()
                trackers[class_id]['tracker'].update(np.array(frame))
                # get the bounding box coordinates from the tracker
                bbox = trackers[class_id]['tracker'].get_position()
                # convert the coordinates to integers
                bbox = [int(coord) for coord in bbox]
                # add the tracker ID to the label
                # labels.append(f"{model.model.names[class_id]} {confidence:0.2f} {trackers[class_id].id}")
                labels.append(f"{model.model.names[class_id]} {confidence:0.2f} {trackers[class_id]['id']}")
            else:
                # initialize a new tracker for this detection
                print(f"xyxy: {xyxy}")
                tracker = cv2.TrackerKCF_create()
                # tracker = cv2.legacy.TrackerMOSSE_create()
                # boundingBox = (xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1])
                boundingBox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1]))
                print(f"boundingBox: {boundingBox}")
                # tracker.init(frame, tuple(xyxy))
                tracker.init(frame, boundingBox)
                trackers[class_id] = tracker
                # add the tracker ID to the label
                tracker_id = str(uuid.uuid4())
                trackers[class_id] = {'tracker': tracker, 'id': tracker_id}
                labels.append(f"{model.model.names[class_id]} {confidence:0.2f} {tracker_id}")
                # labels.append(f"{model.model.names[class_id]} {confidence:0.2f} {tracker.id}")
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        ) 
    
        video.write(frame)
    
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27): # break with escape key
            break
        
            
    cap.release()
    video.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()