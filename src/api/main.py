from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import torch
from src.core.config import settings
from src.core.logger import setup_logger
from src.api.routes import router, set_classifier
from src.models import load_model, ImageClassifier
logger = setup_logger(__name__)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    try:
        device = settings.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            device = "cpu"
        import json
        classes_path = Path(settings.model_path).parent / "classes.json"
        if classes_path.exists():
            with open(classes_path, "r", encoding="utf-8") as f:
                class_names = json.load(f)
        else:
            logger.warning("classes.json not found, using default class names. API inference may be misaligned.")
            class_names = [f"class_{i}" for i in range(10)]
        logger.info(f"Found {len(class_names)} classes")
        model_path = Path(settings.model_path)
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. API will start in degraded mode.")
            set_classifier(None)
            yield
            logger.info("Shutting down...")
            return
        model, _ = load_model(
            num_classes=len(class_names),
            pretrained=False,
            device=device
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Model loaded from {model_path}")
        classifier = ImageClassifier(model, class_names, device=device)
        set_classifier(classifier)
        logger.info("✓ Model ready for inference")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        set_classifier(None)
    yield
    logger.info("Shutting down...")
app = FastAPI(
    title="Image Classification API",
    description="Multi-class image classification using Transfer Learning",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(router, prefix="", tags=["predictions"])
@app.get("/")
async def root():
    return {
        "message": "Image Classification API",
        "docs": "/docs",
        "health": "/health"
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )
