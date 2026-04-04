from fastapi import APIRouter, UploadFile, File, HTTPException, status
from PIL import Image
import io
from src.api.schema import PredictionResponse, HealthResponse
from src.core.config import settings
from src.core.logger import setup_logger
logger = setup_logger(__name__)
router = APIRouter()
_classifier = None
def set_classifier(classifier):
    global _classifier
    _classifier = classifier
def get_classifier():
    return _classifier
@router.get("/health", response_model=HealthResponse)
async def health_check():
    logger.debug("Health check called")
    return {"status": "ok"}
@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile | None = File(default=None)):
    classifier = get_classifier()
    if classifier is None:
        logger.error("Classifier not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid file type. Accept: JPEG, PNG"
        )
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
        max_size = settings.max_upload_size_bytes
        if len(contents) > max_size:
            logger.warning(f"File too large: {len(contents)} bytes (max: {max_size})")
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max: {settings.max_upload_size_mb}MB)"
            )
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = classifier.predict(image)
        logger.info(f"Prediction: {result['predicted_class']} (confidence: {result['confidence']:.4f})")
        return result
    except Image.UnidentifiedImageError:
        logger.error("Cannot decode image")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot decode image"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
