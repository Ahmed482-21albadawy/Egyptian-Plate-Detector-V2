from pydantic import BaseModel


class ImageResponse(BaseModel):

    plate_text : str
    confidence : float