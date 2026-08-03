import os
import cv2
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "best.pt"

INPUT_FOLDER = "dataset_images"     # your images
OUTPUT_FOLDER = "all_debug_plates"  # cropped plates

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

plate_count = 0

# -----------------------------
# PROCESS IMAGES
# -----------------------------
for image_name in os.listdir(INPUT_FOLDER):

    image_path = os.path.join(
        INPUT_FOLDER,
        image_name
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    results = model(image)[0]

    for idx, result in enumerate(
        results.boxes.data.tolist()
    ):

        x1, y1, x2, y2, score, class_id = result

        # Skip weak detections
        if score < 0.30:
            continue

        # Padding
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.25)

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))

        x2 = min(image.shape[1], int(x2 + pad_x))
        y2 = min(image.shape[0], int(y2 + pad_y))

        plate_crop = image[
            y1:y2,
            x1:x2
        ]

        if plate_crop.size == 0:
            continue

        save_path = os.path.join(
            OUTPUT_FOLDER,
            f"plate_{plate_count}.jpg"
        )

        cv2.imwrite(
            save_path,
            plate_crop
        )

        plate_count += 1

    print(
        f"Processed: {image_name}"
    )

print(
    f"\nFinished. Saved {plate_count} plates."
)