from src import IMAGE_FORMATS
from src import SignatureDataset

import re
import random

from collections import defaultdict
from typing import List, Tuple, Dict, Any, DefaultDict
from pathlib import Path

import torch

import numpy as np
from sklearn.model_selection import train_test_split # pyright: ignore[reportUnknownVariableType]

# import seaborn as sns

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    work_seed = 42 + worker_id
    np.random.seed(work_seed)
    random.seed(work_seed)

def extract_signer_id(file_name: str) -> str:
    match = re.search(r"(?:original|forgeries|forgery|forged)_(\d+)_", file_name)
    if match: return match.group(1) 
    return f"UNKNOWN_SIGNER"

def retrieve_signature_images(
    images_path: Path, 
) -> List[Tuple[str, str]]:
    
    signature_images: List[Tuple[str, str]] = []
    if not images_path.is_dir():
        print(f"Warning: Directory not found! {images_path}") 
        return []
    for image_path in images_path.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_FORMATS:
            signer_id = extract_signer_id(str(image_path))
            if signer_id != "UNKNOWN_SIGNER":
                signature_images.append((signer_id, str(image_path)))
            else:
                print(
                    f"Warning: Could not extract signer ID from file: {image_path.name}"
                )
    return signature_images

def prepare_signature_map(
    original_signatures: List[Tuple[str, str]],
    forged_signatures: List[Tuple[str, str]],
) -> Dict[str, Dict[str, List[str]]]:
    
    signature_dictionary: defaultdict[str, Dict[str, List[Any]]] = defaultdict(
        lambda: {
            "original": [], 
            "forged": []
        }
    )
    
    for category, signatures in [("original", original_signatures),
                                 ("forged", forged_signatures)]:
        for signer_id, image_path in signatures:
            signature_dictionary[signer_id][category].append(image_path)
            
    return signature_dictionary

def training_and_testing_split(
    signature_dictionary: Dict[str, Dict[str, List[str]]],
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[
        Dict[str, Dict[str, List[str]]], 
        Dict[str, Dict[str, List[str]]],
        Dict[str, Dict[str, List[str]]]
    ]:
    train_map: Dict[str, Dict[str, List[str]]] = {}
    val_map: Dict[str, Dict[str, List[str]]] = {}
    test_map: Dict[str, Dict[str, List[str]]] = {} 
    
    signer_ids: List[str] = sorted(list(signature_dictionary.keys()), key=int)
    
    strat_labels: List[int] = []
    for sid in signer_ids:
        originals = len(signature_dictionary[sid].get("original", []))
        forgeries = len(signature_dictionary[sid].get("forged", []))
        strat_labels.append(0 if originals >= forgeries else 1)

    train_val_ids, test_ids = train_test_split( # pyright: ignore[reportUnknownVariableType]
        signer_ids,
        test_size=test_ratio,
        random_state=random_state,
        stratify=strat_labels,
    )

    strat_labels_train_val = [
        strat_labels[signer_ids.index(sid)] for sid in train_val_ids  # type: ignore
    ]
    val_size = val_ratio / (1 - test_ratio)
    train_ids, val_ids = train_test_split( # pyright: ignore[reportUnknownVariableType]
        train_val_ids, # pyright: ignore[reportUnknownArgumentType]
        test_size=val_size,
        random_state=random_state,
        stratify=strat_labels_train_val,
    )

    train_map = {sid: signature_dictionary[sid] for sid in train_ids} # pyright: ignore[reportUnknownVariableType]
    val_map = {sid: signature_dictionary[sid] for sid in val_ids} # pyright: ignore[reportUnknownVariableType]
    test_map = {sid: signature_dictionary[sid] for sid in test_ids} # pyright: ignore[reportUnknownVariableType] 

    return train_map, val_map, test_map

def build_label_to_indices(dataset: SignatureDataset) -> dict[str, List[int]]:
    label_to_indices: DefaultDict[str, List[int]] = defaultdict(list)
    for index in range(len(dataset)):
        _, label = dataset[index]
        label_to_indices[label].append(index)
    
    return dict(label_to_indices)

def pretty_metrics(metrics: Dict[str, Any]) -> None:
    print("=== Evaluation Metrics ===")
    print(f"AUC       : {metrics['AUC']:.4f}")
    print(f"EER       : {metrics['EER']:.4f}")
    print(f"Threshold : {metrics['threshold']:.4f}")

    fpr = metrics['fpr']
    tpr = metrics['tpr']
    if len(fpr) > 0:
        print(f"ROC Points: {len(fpr)}")
        print(f"FPR range : {fpr.min():.4f} → {fpr.max():.4f}")
        print(f"TPR range : {tpr.min():.4f} → {tpr.max():.4f}")