import cv2 as cv

import torch
import torchvision.models as models # pyright: ignore[reportMissingTypeStubs]
import torchvision.transforms as transforms # pyright: ignore[reportMissingTypeStubs]

from typing import Dict, Tuple, Any, List, TypedDict
from dataclasses import dataclass

@dataclass
class CLAHEConfig:
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)

@dataclass
class BlurConfig:
    kernel_size: Tuple[int, int] = (5, 5)
    
@dataclass 
class ThresholdConfig:
    method: int = cv.THRESH_BINARY + cv.THRESH_OTSU
    
DATASET_SOURCE: Dict[str, str] = {
    "cedar": "shreelakshmigp/cedardataset", 
}

class SchedulerEntry(TypedDict):
    name: str
    params: Dict[str, Any]

class SchedulerConfig(TypedDict, total=False):
    SCHEDULER: str
    MILESTONES: List[int]
    SCHEDULERS: List[SchedulerEntry]

class ModelState(TypedDict):
    epoch: int
    model_state_dict: Any
    optimiser_state_dict: Any
    scheduler_state_dict: Any
    auc: float
    best_loss: float
    patience_counter: int
    
DATASET_PATH: Dict[str, str] = {
    "CEDAR": "data\\CEDAR"
}

LEARNING_CONFIG: Dict[str, str | int | float] = {
    "BATCH_SIZE": 32,
    "EPOCH": 50,
    "LEARNING_RATE": 1e-3,
    "EARLY_STOPPING_PATIENT": 10,
    
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    "CHECKPOINT_DIR": "checkpoint/exp_01_bh",
    "LOG_DIR": "runs/exp_01_bh"
}

OPTIMISER_PARAMS: Dict[str, str | float] = {
    "optimiser": "AdamW",
    # Prevent overwriting the pretrained weights too aggressively
    "weight_decay": 1e-3,
}

# Linear Warmup and Cosine Decay
SCHEDULER_PARAMS: SchedulerConfig = {
    "SCHEDULER": "SequentialLR", 
    "MILESTONES": [5],
    "SCHEDULERS": [
        {
            "name": "LinearLR",
            "params": {
                "start_factor": 0.1,
                "total_iters": 5
            }
        },
        {
            "name": "CosineAnnealingLR",
            "params": {
                "T_max": int(LEARNING_CONFIG["EPOCH"])-5, 
                "eta_min": 1e-6
            }
        }
    ],
}

BACKBONE_CONFIG: Dict[str, Dict[str, Any]] = {
    "efficientnet_v2_s": {"builder": models.efficientnet_v2_s, "out_channels": 1280},
    "efficientnet_v2_m": {"builder": models.efficientnet_v2_m, "out_channels": 1280},
    "efficientnet_v2_l": {"builder": models.efficientnet_v2_l, "out_channels": 1280},
}

IMAGE_FORMATS: List[str] = [".png", ".jpg", ".jpeg", ".bmp"]

TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomAffine(
            degrees=(-5, 5), 
            translate=(0.1, 0.1), 
            scale=(0.95, 1.05), 
            shear=(-5, 5)
        ),

        transforms.RandomResizedCrop(
            (384, 384), 
            scale=(0.9, 1.05), 
            ratio=(0.95, 1.05), 
            antialias=True
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5], std=[0.5]
    )])

TEST_TRANSFORM = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5], std=[0.5]
    )])