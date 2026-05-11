import cv2
from api.utils import (
    detect_plate,
    crop_plate,
    recognize_characters,
    # process_video,
    map_plate_text,
    draw_detections
)

IMAGE_PATH = r"Custom_test/plate.jpg"


# IMAGE MODE

def run_image(image_path: str):

    # Detect plate
    detections = detect_plate(image_path)

    if not detections:
        print("No plate detected.")
        return

    # Crop plate
    cropped_plates = crop_plate(image_path, detections)

    # Recognize characters
    raw_text = recognize_characters(cropped_plates[0])
    print(f"Raw text: {raw_text}")

    # Map to Arabic
    arabic_text = map_plate_text(raw_text)
    print(f"Arabic text: {arabic_text}")

    # Draw and display
    annotated = draw_detections(image_path, detections, raw_text)
    max_display = 900
    h, w        = annotated.shape[:2]
    if max(h, w) > max_display:
        scale     = max_display / max(h, w)
        annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

    cv2.imshow(f"Plate Text: {raw_text}", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


run_image(IMAGE_PATH)
