"""
video_processor.py

This is your original ALPR script, refactored into a reusable function
so the API can call it. The detection logic itself is UNCHANGED -
only the structure changed (loop wrapped in a function, models loaded
once at import time instead of inside the loop).
"""

from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv, get_best_plate

# ------------------------------------------------------------
# Load models ONCE when this module is imported (i.e. once when
# the API starts up) - NOT inside the function below. Reloading
# a YOLO model on every request would make each request take
# several extra seconds for no reason.
# ------------------------------------------------------------
vehicle_detector = YOLO('yolov8n.pt')
license_plate_detector = YOLO('best.pt')

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


def process_video(video_path: str, output_csv_path: str = None) -> dict:
    """
    Runs the full ALPR pipeline on a video file.

    Args:
        video_path: path to the video file on disk
        output_csv_path: if provided, writes results to this CSV path
                          (same as your original write_csv call)

    Returns:
        results dict shaped like: {frame_number: {car_id: {...}}}
        (identical structure to your original script's `results`)
    """
    # A NEW tracker per video - critical. If this were a global/shared
    # tracker, one user's video would get mixed up with another's.
    mot_tracker = Sort()

    results = {}
    cap = cv2.VideoCapture(video_path)

    frame_nmr = -1
    ret = True

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()

        if ret:
            results[frame_nmr] = {}

            # -------------------------
            # Vehicle Detection
            # -------------------------
            detections = vehicle_detector(frame)[0]
            detections_ = []

            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in VEHICLE_CLASSES:
                    detections_.append([x1, y1, x2, y2, score])

            # -------------------------
            # Vehicle Tracking
            # -------------------------
            track_ids = mot_tracker.update(np.asarray(detections_))

            # -------------------------
            # License Plate Detection
            # -------------------------
            license_plates = license_plate_detector(frame)[0]

            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                if car_id != -1:
                    h, w = frame.shape[:2]
                    pad_x = int((x2 - x1) * 0.15)
                    pad_y = int((y2 - y1) * 0.25)

                    x1_crop = max(0, int(x1 - pad_x))
                    y1_crop = max(0, int(y1 - pad_y))
                    x2_crop = min(w, int(x2 + pad_x))
                    y2_crop = min(h, int(y2 + pad_y))

                    license_plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]

                    license_plate_text, license_plate_text_score = read_license_plate(
                        license_plate_crop
                    )

                    license_plate_text = get_best_plate(car_id, license_plate_text)

                    if license_plate_text is not None:
                        results[frame_nmr][car_id] = {
                            'car': {
                                'bbox': [xcar1, ycar1, xcar2, ycar2]
                            },
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }

    cap.release()

    if output_csv_path:
        write_csv(results, output_csv_path)

    return results
