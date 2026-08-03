import os
import cv2
from paddleocr import PaddleOCR

# Initialize OCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)

debug_folder = "debug_plates"

for image_name in os.listdir(debug_folder):

    image_path = os.path.join(
        debug_folder,
        image_name
    )

    img = cv2.imread(image_path)

    if img is None:
        continue

    result = ocr.ocr(
        image_path,
        cls=True
    )

    print("\n------------------")
    print(image_name)

    if result and result[0]:

        for line in result[0]:

            text = line[1][0]
            score = line[1][1]

            print(
                f"Detected: {text} | Confidence: {score:.2f}"
            )

    else:

        print("No text found")