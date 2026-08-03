import cv2
import pandas as pd
import re

# load results
results = pd.read_csv("test.csv")

# open video
cap = cv2.VideoCapture("sample4.mp4")

# get video properties
fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "alpr_output3.mp4",
    fourcc,
    fps,
    (width, height)
)


def parse_bbox(bbox_string):

    numbers = re.findall(
        r"[-+]?\d*\.\d+|\d+",
        str(bbox_string)
    )

    numbers = [float(n) for n in numbers]

    return numbers[-4:]


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # IMPORTANT
    frame_nmr = int(
        cap.get(cv2.CAP_PROP_POS_FRAMES)
    ) - 1

    frame_data = results[
        results["frame_nmr"] == frame_nmr
    ]

    for _, row in frame_data.iterrows():

        car_bbox = parse_bbox(
            row["car_bbox"]
        )

        plate_bbox = parse_bbox(
            row["license_plate_bbox"]
        )

        if len(car_bbox) != 4:
            continue

        if len(plate_bbox) != 4:
            continue

        x1, y1, x2, y2 = map(
            int,
            car_bbox
        )

        px1, py1, px2, py2 = map(
            int,
            plate_bbox
        )

        plate_text = str(
            row["license_plate_text"]
        )

        # Vehicle box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # Plate box
        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            (0, 0, 255),
            3
        )

        # Black label
        label_height = 40

        label_y = max(
            0,
            py1 - label_height
        )

        cv2.rectangle(
            frame,
            (px1, label_y),
            (px2, py1),
            (0, 0, 0),
            -1
        )

        # White text
        cv2.putText(
            frame,
            plate_text,
            (px1 + 5, py1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    # SAVE FRAME TO OUTPUT VIDEO
    out.write(frame)

cap.release()
out.release()

print("\nVideo saved as: alpr_output3.mp4")