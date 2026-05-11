import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from schemas import ImageResponse
from services import run_image


app = FastAPI()

@app.post("/detect/image", response_model=ImageResponse)
async def detect_image(file: UploadFile = File(...)):
    """Receives a car image, detects the license plate,
       and returns the Arabic plate text and confidence score."""

    # Save uploaded file to a temp path
    tmp_path = None
    try:
        ext = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Run image pipeline
        plate_text, confidence = run_image(tmp_path)

        # Handle no detection
        if plate_text is None:
            raise HTTPException(
                status_code = 404,
                detail = "No license plate detected in the image."
            )

        return ImageResponse(
            plate_text = plate_text,
            confidence = round(confidence, 2)
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)