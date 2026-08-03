import cv2
import os
import shutil

INPUT = "all_debug_plates"
OUTPUT = "good_plates"

os.makedirs(OUTPUT, exist_ok=True)

# -----------------------------
# FILTER PARAMETERS
# -----------------------------

BLUR_THRESHOLD = 100

MIN_WIDTH = 80
MIN_HEIGHT = 25

MIN_RATIO = 2.0
MAX_RATIO = 6.5

saved = 0
rejected = 0

for img_name in os.listdir(INPUT):

    path = os.path.join(INPUT, img_name)

    img = cv2.imread(path)

    if img is None:
        continue

    h, w = img.shape[:2]

    # -----------------------------
    # SIZE FILTER
    # -----------------------------

    if w < MIN_WIDTH or h < MIN_HEIGHT:
        rejected += 1
        continue

    # -----------------------------
    # ASPECT RATIO FILTER
    # -----------------------------

    ratio = w / h

    if ratio < MIN_RATIO or ratio > MAX_RATIO:
        rejected += 1
        continue

    # -----------------------------
    # BLUR FILTER
    # -----------------------------

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if blur_score < BLUR_THRESHOLD:
        rejected += 1
        continue

    # -----------------------------
    # KEEP IMAGE
    # -----------------------------

    shutil.copy(
        path,
        os.path.join(
            OUTPUT,
            img_name
        )
    )

    saved += 1

print("\nFinished")
print(f"Saved    : {saved}")
print(f"Rejected : {rejected}")