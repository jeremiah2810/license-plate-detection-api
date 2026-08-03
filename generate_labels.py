import os
import csv
import cv2

from util import read_license_plate

INPUT_FOLDER = "good_plates"

with open("labels.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "image",
        "predicted_text"
    ])

    total = 0
    detected = 0

    for image_name in os.listdir(INPUT_FOLDER):

        image_path = os.path.join(
            INPUT_FOLDER,
            image_name
        )

        img = cv2.imread(image_path)

        if img is None:
            continue

        total += 1

        text, score = read_license_plate(img)

        if text is None:
            text = ""

        else:
            detected += 1

        writer.writerow([
            image_name,
            text
        ])

        print(
            f"{image_name} -> {text}"
        )

print("\nDone")
print(f"Total images    : {total}")
print(f"OCR detections  : {detected}")
print("Saved labels.csv")