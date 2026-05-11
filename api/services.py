from utils import (
    detect_plate,
    crop_plate,
    recognize_characters,
    map_plate_text,
)


def run_image(image_path: str):

    # Detect plate
    detections = detect_plate(image_path)

    if not detections:
        return None, None

    # Crop plate
    cropped_plates = crop_plate(image_path, detections)

    # Recognize characters
    raw_text = recognize_characters(cropped_plates[0])

    # Map to Arabic
    arabic_text = map_plate_text(raw_text)

    # Get confidence of best detection
    confidence = max(detections, key=lambda d: d["confidence"])["confidence"]

    return arabic_text, confidence