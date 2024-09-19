import torch
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import time 


torch.cuda.set_device(0) # gpu 0

weights = 'best_colab.pt'
model = YOLO(weights)

cap = cv2.VideoCapture(0)  # camera 0

track_history = defaultdict(lambda: [])  
missed_detection_counter = defaultdict(int) 
id_history = set()  

MISS_THRESHOLD = 10  # how many frames are allowed to be missed 
WAVE_DURATION = 5
PALM_CLASS_ID = 1  
MIN_MOVEMENT_THRESHOLD_FRACTION = 0.05  # for micromovement artifacts, how much of bbox to travel in 1 frame


def analyze_wave(track, min_movement_threshold):
    if len(track) <  WAVE_DURATION * 2:  # have frames for at least 2 movements
        return False
    

    x_coords = [coord[0] for coord in track[-30:]] 
    widths = [coord[2] for coord in track[-30:]] 

    
    avg_width = sum(widths) / len(widths) if widths else 0   # calculate average bbox width
    
    direction_changes = 0
    consistent_movement = 0
    last_direction = 0
    
    for i in range(1, len(x_coords)):
        movement = x_coords[i] - x_coords[i-1]
        direction = np.sign(movement)
        
        
        if abs(movement) > avg_width * min_movement_threshold:  # if movement was big enough
            if direction != last_direction:
                if consistent_movement >= WAVE_DURATION:
                    direction_changes += 1
                consistent_movement = 1  # reset for new direction
                last_direction = direction
            else:
                consistent_movement += 1
        
    return direction_changes >= 2 # 2 changes


while cap.isOpened():
    frame_start_time = time.time() 
    success, frame = cap.read()
    if not success:
        break 

    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()

    current_ids = set()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            if class_id == PALM_CLASS_ID: 
                current_ids.add(track_id)
                missed_detection_counter[track_id] = 0

                x, y, w, h = box
                track = track_history[track_id]
                track.append((x , y, w))

                if analyze_wave(track, MIN_MOVEMENT_THRESHOLD_FRACTION):
                    text_position = (x, y + h // 2)
                    cv2.putText(annotated_frame, f"Wave ID: {track_id}", text_position, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


    # increment missed detections, to combat skipped frames of no detection
    for track_id in id_history - current_ids:
        missed_detection_counter[track_id] += 1
        if missed_detection_counter[track_id] > MISS_THRESHOLD:
            track_history[track_id].clear()

    id_history.update(current_ids)


    frame_end_time = time.time()
    fps = 1 / (frame_end_time - frame_start_time)

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (annotated_frame.shape[1] - 150, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Handwave recognition", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
