import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Union, cast
from torchvision.transforms import ToTensor
from src.utils.transforms import get_val_transforms
from src.core.logger import setup_logger
logger = setup_logger(__name__)
class ImageClassifier:
    def __init__(
        self,
        model: torch.nn.Module,
        class_names: List[str],
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.class_names = class_names
        self.transforms = get_val_transforms()
    @torch.no_grad()
    def predict(self, image_input: Union[Image.Image, str, Path]) -> Dict[str, Any]:
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Input must be PIL Image or path")
        transformed = self.transforms(image)
        if isinstance(transformed, torch.Tensor):
            tensor = transformed
        else:
            tensor = cast(torch.Tensor, ToTensor()(transformed))
        image_tensor = tensor.unsqueeze(0).to(self.device)
        logits = self.model(image_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
        predicted_class = self.class_names[int(pred_idx.item())]
        confidence = confidence.item()
        logger.debug(f"Prediction: {predicted_class} ({confidence:.4f})")
        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence)
        }
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(img_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for {img_path}: {e}")
                results.append({"error": str(e)})
        return results
