import sys
from pathlib import Path
import shutil
from typing import Union
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchvision.datasets as datasets
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import settings
from src.core.logger import setup_logger
logger = setup_logger(__name__)
def download_and_prepare_dataset(
    data_path: Union[str, Path] = "data/",
    train_split: float = 0.8,
    random_seed: int = 42
) -> None:
    logger.info("Starting dataset download and preparation...")
    data_root = Path(data_path)
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if train_dir.exists() and val_dir.exists():
        train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
        val_classes = [d for d in val_dir.iterdir() if d.is_dir()]
        if train_classes and val_classes:
            logger.info("Dataset already prepared!")
            return
    if train_dir.exists() or val_dir.exists():
        logger.info("Found partial dataset structure. Rebuilding train/val splits.")
    logger.info("Downloading Caltech-101 dataset...")
    temp_dir = data_root.parent / "temp_caltech"
    temp_dir.mkdir(exist_ok=True)
    try:
        dataset = datasets.Caltech101(root=str(temp_dir), download=True)
        logger.info(f"Downloaded {len(dataset)} images from Caltech-101")
        for d in [train_dir, val_dir]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        class_samples = {}
        categories_root = temp_dir / "caltech101" / "101_ObjectCategories"
        for image_idx, label in tqdm(zip(dataset.index, dataset.y), total=len(dataset), desc="Indexing"):
            class_name = dataset.categories[label]
            img_path = categories_root / class_name / f"image_{int(image_idx):04d}.jpg"
            if not img_path.exists():
                logger.warning(f"Skipping missing image path: {img_path}")
                continue
            if class_name not in class_samples:
                class_samples[class_name] = []
            class_samples[class_name].append((str(img_path), label, class_name))
        total_train = 0
        total_val = 0
        for class_name, samples in tqdm(class_samples.items(), desc="Organizing"):
            if len(samples) < 2:
                logger.warning(f"Skipping class '{class_name}' with < 2 samples")
                continue
            indices = list(range(len(samples)))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=1 - train_split,
                random_state=random_seed
            )
            train_class_dir = train_dir / class_name
            train_class_dir.mkdir(parents=True, exist_ok=True)
            for i in train_idx:
                src = samples[i][0]
                dst = train_class_dir / Path(src).name
                shutil.copy2(src, dst)
                total_train += 1
            val_class_dir = val_dir / class_name
            val_class_dir.mkdir(parents=True, exist_ok=True)
            for i in val_idx:
                src = samples[i][0]
                dst = val_class_dir / Path(src).name
                shutil.copy2(src, dst)
                total_val += 1
        logger.info(f"Train set: {total_train} images")
        logger.info(f"Val set: {total_val} images")
        shutil.rmtree(temp_dir)
        logger.info("Dataset preparation complete!")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
if __name__ == "__main__":
    download_and_prepare_dataset(
        data_path=settings.data_path,
        train_split=settings.train_split,
        random_seed=settings.random_seed
    )
