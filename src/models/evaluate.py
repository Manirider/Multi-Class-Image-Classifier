import argparse
import torch
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union, cast
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import numpy as np
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.logger import setup_logger
from src.core.config import settings
from src.data.dataset import create_dataloaders
from src.models.model import load_model
from src.utils import get_val_transforms
logger = setup_logger(__name__)
@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    class_names: List[str]
) -> Tuple[Dict[str, Any], np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []
    logger.info("Evaluating model...")
    if len(val_loader) == 0:
        raise ValueError("Validation loader is empty. Ensure data/val has valid images.")
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    report = cast(Dict[str, Dict[str, Any]], classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    ))
    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "confusion_matrix": conf_matrix.tolist(),
        "per_class_metrics": {
            class_name: {
                "precision": float(report[class_name]["precision"]),
                "recall": float(report[class_name]["recall"]),
                "f1_score": float(report[class_name]["f1-score"]),
                "support": int(report[class_name]["support"])
            }
            for class_name in class_names
        }
    }
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    return metrics, all_preds
def save_metrics(metrics: Dict[str, Any], output_path: Union[str, Path]) -> None:
    metrics_path = Path(output_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--device", default=settings.device, help="Device: cuda or cpu")
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    try:
        _, val_loader, num_classes = create_dataloaders(
            data_path=settings.data_path,
            batch_size=settings.batch_size,
            val_batch_size=settings.val_batch_size,
            num_workers=settings.num_workers,
            train_transform=get_val_transforms(settings.image_size),
            val_transform=get_val_transforms(settings.image_size),
        )
        model_path = Path(settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model, _ = load_model(num_classes=num_classes, pretrained=False, device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        import json
        classes_path = Path(settings.model_path).parent / "classes.json"
        if not classes_path.exists():
            raise FileNotFoundError(f"Missing classes.json at {classes_path}")
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
        metrics, _ = evaluate_model(
            model=model,
            val_loader=val_loader,
            device=device,
            class_names=val_dataset.classes,
        )
        save_metrics(metrics, str(Path(settings.results_dir) / "metrics.json"))
        return 0
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}", exc_info=True)
        return 1
if __name__ == "__main__":
    raise SystemExit(main())
