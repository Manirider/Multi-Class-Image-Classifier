from pathlib import Path
from typing import Tuple, Optional, cast
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from PIL import Image, UnidentifiedImageError
from src.core.logger import setup_logger
logger = setup_logger(__name__)
class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Compose] = None,
        class_to_idx: Optional[dict] = None
    ):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = class_to_idx or {}
        self.idx_to_class = {}
        self._load_samples()
    @property
    def classes(self) -> list[str]:
        return [self.idx_to_class[i] for i in range(len(self.class_to_idx))]
    def _load_samples(self):
        for class_idx, class_dir in enumerate(sorted(self.root.iterdir())):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name
            for img_file in class_dir.glob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif"]:
                    self.samples.append((str(img_file), self.class_to_idx[class_name]))
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.class_to_idx)} classes")
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                transformed = self.transform(image)
                if isinstance(transformed, torch.Tensor):
                    tensor = transformed
                else:
                    tensor = cast(torch.Tensor, ToTensor()(transformed))
            else:
                tensor = ToTensor()(image)
            return tensor, label
        except (UnidentifiedImageError, OSError) as exc:
            logger.warning(f"Skipping unreadable image {img_path}: {exc}")
            fallback_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(fallback_idx)
def create_dataloaders(
    data_path: str,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
    train_transform: Compose,
    val_transform: Compose
) -> Tuple[DataLoader, DataLoader, int]:
    train_dir = Path(data_path) / "train"
    val_dir = Path(data_path) / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Dataset directories not found at {data_path}")
    num_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
    if num_classes == 0:
        raise ValueError(f"No class subdirectories found in {train_dir}")
    logger.info(f"Number of classes: {num_classes}")
    train_dataset = ImageFolderDataset(str(train_dir), transform=train_transform)
    val_dataset = ImageFolderDataset(str(val_dir), transform=val_transform)
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    if len(train_dataset) == 0:
        raise ValueError(f"No training images found under {train_dir}")
    if len(val_dataset) == 0:
        raise ValueError(f"No validation images found under {val_dir}")
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, num_classes
