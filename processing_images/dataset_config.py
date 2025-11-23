import cv2 as cv
from typing import Dict, Tuple
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