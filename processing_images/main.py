from process_images import apply_threshold, read_images, check_image_type, write_images
from process_images import CLAHEConfig, BlurConfig, ThresholdConfig

import numpy as np
import numpy.typing as npt
from pathlib import Path
from itertools import chain
import sys

def has_output_folder(folder: Path,):
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Could not create output folder {folder}: {e}", file=sys.stderr)
        
def threshold_image(
    image_path: Path, 
    clahe_params: CLAHEConfig,
    blur_params: BlurConfig,
    threshold_params: ThresholdConfig,
) -> npt.NDArray[np.uint8]:
    raw_image = check_image_type(read_images(image_path))
    thresholded_image = apply_threshold(
        raw_image,
        clahe_params,
        blur_params,
        threshold_params
    )
    return thresholded_image
    
def image_processing_pipeline(
    raw_image_folder: Path,
    thresholed_image_folder: Path,
    clahe_params: CLAHEConfig,
    blur_params: BlurConfig,
    threshold_params: ThresholdConfig
) -> None:
    image_formats = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"]
    all_images = chain.from_iterable(raw_image_folder.glob(p) for p in image_formats)
    for image_path in all_images:
        thresholded_image = threshold_image(
            image_path, 
            clahe_params,
            blur_params,
            threshold_params
        )
        
        if thresholded_image.size > 0:
            output_path = thresholed_image_folder / image_path.name
            write_images(output_path, thresholded_image)
        else: print(f"Skipped {image_path.name} due to internal errors.", file=sys.stderr)
        
if __name__ == '__main__':
    clahe_params = CLAHEConfig()
    blur_params = BlurConfig()
    threshold_params = ThresholdConfig()
    
    # Assume you are calling from the command line
    dataset_source = "CEDAR"
    
    raw_forged_folder = Path(f"raw_signature_images/{dataset_source}/forged")
    raw_original_folder = Path(f"raw_signature_images/{dataset_source}/original")
    
    output_forged_folder = Path(f"data/{dataset_source}/forged")
    output_original_folder = Path(f"data/{dataset_source}/original")
    
    output_forged_folder.mkdir(parents=True, exist_ok=True)
    output_original_folder.mkdir(parents=True, exist_ok=True)
    
    image_processing_pipeline(
        raw_forged_folder,
        output_forged_folder,
        clahe_params,
        blur_params,
        threshold_params
    )
    
    image_processing_pipeline(
        raw_original_folder,
        output_original_folder,
        clahe_params,
        blur_params,
        threshold_params,
    )