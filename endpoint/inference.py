import torch
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import time
import argparse
import boto3
from flask import Flask, request
import flask
import logging
import os
import json

app = Flask(__name__)

weights = '/opt/ml/model/best.pt'  # where weights are being copied
# weights = 'best.pt'
model = YOLO(weights)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@app.get('/ping')
def ping():
    health = model is not None
    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.post('/invocations')
def invoke():
    input_json = flask.request.get_json()
    input_path = input_json['input_path']
    output_path = input_json['output_path']
    print(input_path, output_path)
    resp = inference(input_path, output_path)
    result = {'output': resp}
    resultjson = json.dumps(result)
    return flask.Response(response=resultjson, status=200, mimetype='application/json')


def inference(input_path_s3, output_path_s3):
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        logging.info(f"Using gpu ${torch.cuda.current_device()}")
    else:
        logging.info("No GPU found")

    # parser = argparse.ArgumentParser(description="Run YOLOv8 tracking on a video.")
    # parser.add_argument("input_path", type=str, help="Path to video in S3 bucket")
    # parser.add_argument("output_path", type=str, help="Path to save the output in S3")

    # parser.add_argument("output_path", nargs='?', default=None, type=str,
    #                     help="Optional: Path to save the output video")

    # args = parser.parse_args()

    bucket_name = 'sagemaker-us-east-1-dpetrovic'
    session = boto3.Session()
    s3_client = session.client('s3')

    # input_path_s3 = args.input_path
    app_input = "/tmp/"
    input_path_local = app_input + input_path_s3.split('/')[-1]

    try:
        s3_client.download_file(bucket_name, input_path_s3, input_path_local)
        logging.info(f"Downloaded file to {input_path_local}")
    except Exception as e:
        logging.error(f'Error downloading input: {e.args}')

    cap = cv2.VideoCapture(input_path_local)  # camera 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # output_path_s3 = args.output_path
    output_path_local = app_input + output_path_s3.split('/')[-1]

    logging.info(f"Input local path = {input_path_local}, output local path {output_path_local}")

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path_local, fourcc, fps, (frame_width, frame_height))

    track_history = defaultdict(lambda: [])
    missed_detection_counter = defaultdict(int)
    id_history = set()

    MISS_THRESHOLD = 5  # how many frames are allowed to be missed
    WAVE_DURATION = 3
    PALM_CLASS_ID = 1
    MIN_MOVEMENT_THRESHOLD_FRACTION = 0.05  # for micromovement artifacts, how much of bbox to travel in 1 frame

    def analyze_wave(track, min_movement_threshold):
        """Analyze the track to detect wave based on X movement.
           Only considers movement as significant if it exceeds the min_movement_threshold."""
        if len(track) < WAVE_DURATION * 2:  # have frames for at least 2 movements
            return False

        x_coords = [coord[0] for coord in track[-30:]]
        widths = [coord[2] for coord in track[-30:]]

        avg_width = sum(widths) / len(widths) if widths else 0  # calculate average bbox width

        direction_changes = 0
        consistent_movement = 0
        last_direction = 0

        for i in range(1, len(x_coords)):
            movement = x_coords[i] - x_coords[i - 1]
            direction = np.sign(movement)

            if abs(movement) > avg_width * min_movement_threshold:  # if movement was big enough
                if direction != last_direction:
                    if consistent_movement >= WAVE_DURATION:
                        direction_changes += 1
                    consistent_movement = 1  # reset for new direction
                    last_direction = direction
                else:
                    consistent_movement += 1

        return direction_changes >= 2  # 2 changes

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
                    track.append((x, y, w))

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
        if output_path_s3:
            out.write(annotated_frame)
        else:
            cv2.imshow("Handwave recognition", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    try:
        s3_client.upload_file(output_path_local, bucket_name, output_path_s3)
        logging.info(f'Saved output to {output_path_s3}')
    except Exception as e:
        logging.error(f'Error saving output: {e.args}')

    # os.remove(input_path_local)
    # os.remove(output_path_local)

    return f"Result saved at {bucket_name}/{output_path_s3}"


if __name__ == '__main__':
    app.run(port=8080)
