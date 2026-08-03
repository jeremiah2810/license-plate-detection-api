import os
import cv2
import easyocr

reader = easyocr.Reader(['en'])

debug_folder = "debug_plates"

for image_name in os.listdir(debug_folder):

    image_path = os.path.join(
        debug_folder,
        image_name
    )

    img = cv2.imread(image_path)

    result = reader.readtext(img)

    print("\n------------------")
    print(image_name)

    for detection in result:

        text = detection[1]
        score = detection[2]

        print(
            f"Detected: {text} | Confidence: {score:.2f}"
        )