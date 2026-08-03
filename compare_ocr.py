import os
import cv2
import easyocr
from paddleocr import PaddleOCR

# EasyOCR
easy_reader = easyocr.Reader(['en'], gpu=False)

# PaddleOCR
paddle_reader = PaddleOCR(
    use_textline_orientation=True,
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

    print("\n" + "=" * 60)
    print(image_name)

    # -------------------
    # EasyOCR
    # -------------------
    easy_result = easy_reader.readtext(img)

    easy_text = "None"

    if len(easy_result) > 0:
        easy_text = easy_result[0][1]

    print("EasyOCR :", easy_text)

    # -------------------
    # PaddleOCR
    # -------------------
    paddle_text = "None"

    try:

        result = paddle_reader.predict(image_path)

        texts = []

        for res in result:

            if "rec_text" in str(res):
                texts.append(str(res))

        paddle_text = texts

    except Exception as e:

        paddle_text = str(e)

    print("PaddleOCR:", paddle_text)