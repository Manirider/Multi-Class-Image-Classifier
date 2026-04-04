import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Union
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import settings
from src.core.logger import setup_logger
from src.models.model import unfreeze_backbone
from src.models.model import load_model
from src.data.preprocess import download_and_prepare_dataset
from src.data.dataset import create_dataloaders
from src.utils import get_train_transforms, get_val_transforms
logger = setup_logger(__name__)
class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty. Ensure data/train has valid images.")
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    if len(val_loader) == 0:
        raise ValueError("Validation loader is empty. Ensure data/val has valid images.")
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    learning_rate: float = 0.001,
    device: str = "cuda",
    model_save_path: Union[str, Path] = "model/image_classifier.pth"
) -> nn.Module:
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    early_stopping = EarlyStopping(patience=settings.early_stopping_patience)
    best_val_loss = float('inf')
    model_path = Path(model_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            logger.info(f"  ✓ Saved best model to {model_path}")
        scheduler.step()
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    logger.info(f"Loading best model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    return model
def main() -> int:
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=settings.batch_size, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=settings.learning_rate, help="Learning rate")
    parser.add_argument("--device", default=settings.device, help="Device: cuda or cpu")
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    try:
        download_and_prepare_dataset(
            data_path=settings.data_path,
            train_split=settings.train_split,
            random_seed=settings.random_seed,
        )
        train_loader, val_loader, num_classes = create_dataloaders(
            data_path=settings.data_path,
            batch_size=args.batch_size,
            val_batch_size=settings.val_batch_size,
            num_workers=settings.num_workers,
            train_transform=get_train_transforms(settings.image_size),
            val_transform=get_val_transforms(settings.image_size),
        )
        model, _ = load_model(
            num_classes=num_classes,
            pretrained=settings.pretrained,
            device=device,
        )
        import json
        classes_path = Path(settings.model_path).parent / "classes.json"
        with open(classes_path, "w", encoding="utf-8") as f:
            json.dump(train_loader.dataset.classes, f, indent=2)
        logger.info(f"Saved class mapping to {classes_path}")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
            model_save_path=settings.model_path,
        )
        return 0
    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        return 1
if __name__ == "__main__":
    raise SystemExit(main())
