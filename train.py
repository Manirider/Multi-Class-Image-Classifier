import argparse
import torch
from pathlib import Path
from torchvision.datasets import ImageFolder
from src.core.config import settings
from src.core.logger import setup_logger
from src.data.preprocess import download_and_prepare_dataset
from src.data.dataset import create_dataloaders
from src.utils import get_train_transforms, get_val_transforms
from src.models import load_model, train_model
logger = setup_logger(__name__)
def main():
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()
    logger.info("=" * 80)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 80)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}")
    logger.info("\n[1/4] Preparing dataset...")
    try:
        download_and_prepare_dataset(
            data_path=settings.data_path,
            train_split=settings.train_split,
            random_seed=settings.random_seed
        )
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return
    logger.info("\n[2/4] Creating dataloaders...")
    try:
        train_loader, val_loader, num_classes = create_dataloaders(
            data_path=settings.data_path,
            batch_size=args.batch_size,
            val_batch_size=settings.val_batch_size,
            num_workers=settings.num_workers,
            train_transform=get_train_transforms(settings.image_size),
            val_transform=get_val_transforms(settings.image_size)
        )
    except Exception as e:
        logger.error(f"Dataloader creation failed: {e}")
        return
    logger.info("\n[3/4] Loading model...")
    model, _ = load_model(
        num_classes=num_classes,
        pretrained=settings.pretrained,
        device=device
    )
    import json
    classes_path = Path(settings.model_path).parent / "classes.json"
    if not isinstance(train_loader.dataset, ImageFolder):
        raise TypeError("Expected ImageFolder dataset to save class mapping")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(train_loader.dataset.classes, f, indent=2)
    logger.info(f"Saved class mapping to {classes_path}")
    logger.info("\n[4/4] Training model...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        model_save_path=settings.model_path
    )
    logger.info("\n" + "=" * 80)
    logger.info("[OK] TRAINING COMPLETE")
    logger.info("=" * 80)
if __name__ == "__main__":
    main()
