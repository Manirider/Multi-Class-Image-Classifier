import argparse
import torch
from pathlib import Path
from src.core.config import settings
from src.core.logger import setup_logger
from src.data.dataset import create_dataloaders
from src.utils import get_val_transforms
from src.models import load_model, evaluate_model, save_metrics
logger = setup_logger(__name__)
def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()
    logger.info("=" * 80)
    logger.info("EVALUATION PIPELINE")
    logger.info("=" * 80)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}")
    logger.info("\nLoading validation dataset...")
    try:
        _, val_loader, num_classes = create_dataloaders(
            data_path=settings.data_path,
            batch_size=settings.batch_size,
            val_batch_size=settings.val_batch_size,
            num_workers=settings.num_workers,
            train_transform=get_val_transforms(settings.image_size),
            val_transform=get_val_transforms(settings.image_size)
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    logger.info("Loading model...")
    model_path = Path(settings.model_path)
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    model, _ = load_model(
        num_classes=num_classes,
        pretrained=False,
        device=device
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info(f"Model loaded from {model_path}")
    import json
    classes_path = Path(settings.model_path).parent / "classes.json"
    if not classes_path.exists():
        logger.error(f"classes.json not found at {classes_path}. Please train first.")
        return
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    logger.info("\nEvaluating model...")
    metrics, _ = evaluate_model(
        model=model,
        val_loader=val_loader,
        device=device,
        class_names=class_names
    )
    metrics_path = Path(settings.results_dir) / "metrics.json"
    save_metrics(metrics, str(metrics_path))
    logger.info("\n" + "=" * 80)
    logger.info("✓ EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Metrics saved to: {metrics_path}")
if __name__ == "__main__":
    main()
