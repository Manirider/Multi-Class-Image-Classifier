from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field
class Settings(BaseSettings):
    model_path: str = Field(default="model/image_classifier.pth", description="Path to model")
    model_name: str = Field(default="resnet50", description="Model architecture")
    pretrained: bool = Field(default=True, description="Load pretrained weights")
    num_classes: int = Field(default=10, description="Number of output classes")
    device: Literal["cuda", "cpu"] = Field(default="cuda", description="Device type")
    batch_size: int = Field(default=32, description="Training batch size")
    val_batch_size: int = Field(default=64, description="Validation batch size")
    epochs: int = Field(default=20, description="Number of training epochs")
    learning_rate: float = Field(default=0.001, description="Initial learning rate")
    weight_decay: float = Field(default=1e-4, description="Weight decay for regularization")
    early_stopping_patience: int = Field(default=5, description="Early stopping patience")
    num_workers: int = Field(default=4, description="DataLoader workers")
    data_path: str = Field(default="data/", description="Path to dataset")
    train_split: float = Field(default=0.8, description="Train/val split ratio")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    image_size: int = Field(default=224, description="Input image size")
    results_dir: str = Field(default="results/", description="Results directory")
    log_file: str = Field(default="results/training.log", description="Log file path")
    api_port: int = Field(default=8000, description="API port")
    api_host: str = Field(default="0.0.0.0", description="API host")
    max_upload_size_mb: int = Field(default=10, description="Max upload size in MB")
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    def __init__(self, **data):
        super().__init__(**data)
        self._setup_directories()
    def _setup_directories(self):
        directories = [
            Path(self.results_dir),
            Path(self.data_path) / "train",
            Path(self.data_path) / "val",
            Path("model"),
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024
settings = Settings()
