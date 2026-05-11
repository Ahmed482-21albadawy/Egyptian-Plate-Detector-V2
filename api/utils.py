import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


BASE_DIR     = Path(__file__).resolve().parent.parent
model_plate  = YOLO(str(BASE_DIR / "Models" / "plate_detector.pt"))
model_ocr    = YOLO(str(BASE_DIR / "Models" / "char_recognizer.pt"))

# Function 1: Detect plate in a car image

def detect_plate(image_path: str):
    """
    Takes a car image path and returns the detected plate coordinates.

    Args:
        image_path (str): Path to the input car image.

    Returns:
        detections (list): List of dicts with box coords and confidence.
                           Empty list if no plate detected.
    """

    # # ── Load models  ────────────────────────────
    # model_plate = YOLO("Models/plate_detector.pt")
    

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from: {image_path}")

    # Run Model
    results = model_plate.predict(
        source  = image_path,
        conf    = 0.25,
        verbose = False,
        device  = "cpu"
    )

    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence      = box.conf[0].item()

        detections.append({
            "x1"        : x1,
            "y1"        : y1,
            "x2"        : x2,
            "y2"        : y2,
            "confidence": confidence
        })

    return detections


# Function 2: Crop the detected plate from the original image

def crop_plate(image_path: str, detections: list):
    """
    Crops the detected plate region from the original image.

    Args:
        image_path (str) : Path to the original car image.
        detections (list): List of dicts with x1,y1,x2,y2 from detect_plate().

    Returns:
        cropped_plates (list): List of cropped plate images as numpy arrays.
    """

    # Load original image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image from: {image_path}")

    cropped_plates = []

    # Select the widest detection
    if detections:
        best = max(detections, key=lambda d: d["x2"] - d["x1"])
        x1, y1, x2, y2 = best["x1"], best["y1"], best["x2"], best["y2"]

        padding      = 2
        width_margin = 5
        h_img, w_img = original.shape[:2]

        x1 = max(0, x1 - width_margin)
        y1 = max(0, y1 + padding)
        x2 = min(w_img, x2 + width_margin)
        y2 = max(0, y2 - padding)

        cropped = original[y1:y2, x1:x2]
        cropped_plates.append(cropped)

    return cropped_plates


# Function 3: Recognize Arabic characters on the cropped plate

def recognize_characters(cropped_plate):
    """
    Takes a cropped plate image and returns the recognized
    characters sorted by position (left to right).

    Args:
        cropped_plate (numpy array): Cropped plate from crop_plate().

    Returns:
        plate_text (str): Full plate text in correct order.
    """
    # model_ocr   = YOLO("models/char_recognizer.pt")

    result = model_ocr.predict(
        source  = cropped_plate,
        conf    = 0.25,
        verbose = False,
        device  = "cpu"
    )

    plate_chars = []

    for box in result[0].boxes:
        class_id        = int(box.cls)
        recognized_text = result[0].names[class_id]
        x_center        = box.xywh[0][0].item()

        plate_chars.append({
            "text"    : recognized_text,
            "x_center": x_center
        })

    # Sort characters left to right by x position
    plate_chars = sorted(plate_chars, key=lambda c: c["x_center"])

    # Reconstruct full plate text
    plate_text = " ".join([c["text"] for c in plate_chars])

    return plate_text


# Function 4: Map recognized text to Arabic script

# Letter name → Arabic script
PLATE_MAP = {
    "alif" : "ا",
    "baa" : "ب",
    "taa" : "ت",
    "thaa" : "ث",
    "jeem" : "ج",
    "haa" : "ح",
    "khaa" : "خ",
    "daal" : "د",
    "zaal" : "ذ",
    "raa" : "ر",
    "zay" : "ز",
    "seen" : "س",
    "sheen" : "ش",
    "saad" : "ص",
    "daad" : "ض",
    "Taa" : "ط",
    "Thaa" : "ظ",
    "ain" : "ع",
    "ghayn" : "غ",
    "faa" : "ف",
    "qaaf" : "ق",
    "kaaf" : "ك",
    "laam" : "ل",
    "meem" : "م",
    "noon" : "ن",
    "waw" : "و",
    "yaa" : "ي",
    "7aa" : "ة",
    "0" : "٠",
    "1" : "١",
    "2" : "٢",
    "3" : "٣",
    "4" : "٤",
    "5" : "٥",
    "6" : "٦",
    "7" : "٧",
    "8" : "٨",
    "9" : "٩",
}

def map_plate_text(plate_text: str):
    """
    Maps the raw YOLO output text to proper Arabic script
    and reverses the order to match Arabic right-to-left reading.

    Args:
        plate_text (str): Raw plate text from recognize_characters().
                          e.g. "6 6 6 6 qaaf waw raa"

    Returns:
        arabic_text (str): Mapped Arabic text in correct order.
                           e.g. "ر و ق ٦ ٦ ٦ ٦"
    """

    if not plate_text:
        return ""

    # Split into tokens
    tokens = plate_text.strip().split()

    # Map each token, keep as is if not found
    mapped = [PLATE_MAP.get(token, token) for token in tokens]
    # mapped.reverse()

    arabic_text = " ".join(mapped)
    return arabic_text


# Function 5: Draw detections and plate text on the original image

def draw_detections(image_path: str, detections: list, plate_text: str):
    """
    Draws bounding boxes and Arabic plate text on the original image.

    Args:
        image_path  (str) : Path to the original car image.
        detections  (list): List of dicts with x1,y1,x2,y2 from detect_plate().
        plate_text  (str) : Arabic plate text from map_plate_text().

    Returns:
        image (numpy array): Annotated image with bounding box and plate text.
    """

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from: {image_path}")

    h, w = image.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        confidence      = det["confidence"]

        # Bounding box
        color = (0, 200, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label: Arabic text + confidence
        label      = f"{plate_text}  {confidence:.0%}"
        font_scale = max(0.5, min(w, h) / 1000)
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Background pill
        cv2.rectangle(image,
                      (x1, y1 - th - baseline - 8),
                      (x1 + tw + 8, y1),
                      color, -1)

        cv2.putText(image, label, (x1 + 4, y1 - baseline - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), thickness, cv2.LINE_AA)

    return image