import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Dict, Any
from src.core.logger import setup_logger
logger = setup_logger(__name__)
def load_model(
    num_classes: int,
    pretrained: bool = True,
    device: str = "cuda"
) -> Tuple[nn.Module, nn.Module]:
    logger.info(f"Loading ResNet50 (pretrained={pretrained})")
    try:
        if pretrained:
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            base_model = models.resnet50(weights=None)
    except Exception as exc:
        logger.warning(f"Falling back to uninitialized weights: {exc}")
        base_model = models.resnet50(weights=None)
    for param in base_model.parameters():
        param.requires_grad = False
    in_features = base_model.fc.in_features
    base_model.fc = nn.Linear(in_features, num_classes)
    for param in base_model.fc.parameters():
        param.requires_grad = True
    model = base_model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    return model, base_model
def unfreeze_backbone(model: nn.Module, unfreeze_layers: int = 1) -> None:
    layers_to_unfreeze = {
        1: ["layer4"],
        2: ["layer4", "layer3"],
        3: ["layer4", "layer3", "layer2"],
        4: ["layer4", "layer3", "layer2", "layer1"],
    }
    layer_names = layers_to_unfreeze.get(unfreeze_layers, [])
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Unfroze {unfreeze_layers} layer(s). Trainable params: {trainable:,}")
def get_model_size(model: nn.Module) -> float:
    param_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
    buffer_size = sum(b.numel() for b in model.buffers()) * 4 / (1024 ** 2)
    return param_size + buffer_size
