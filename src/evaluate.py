from typing import Dict, Tuple, Any, List
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve # pyright: ignore[reportUnknownVariableType]

def evaluate(
    model: nn.Module, 
    test_loader: DataLoader[Tuple[torch.Tensor, str]], 
    test_map:  Dict[str, Dict[str, List[str]]],
    device: torch.device
) -> Dict[str, Any]:
    model.to(device)
    
    embeddings, labels = embed_all(model, test_loader, device)

    # original, easy, and hard forgery  
    pos_pairs, intra_pairs, _inter_pairs = build_verification_pairs(embeddings, test_map)

    neg_pairs = intra_pairs
    
    # Compute similarities
    pos_scores = compute_scores(embeddings, pos_pairs)
    neg_scores = compute_scores(embeddings, neg_pairs)
    print(f"Num positive pairs: {len(pos_scores)}")
    print(f"Num negative pairs: {len(neg_scores)}")
    
    metrics = compute_metrics(pos_scores, neg_scores)
    operating_point = summarise_operating_point(
        metrics["fpr"], metrics["tpr"], thresholds = metrics["thresholds"], chosen_threshold = metrics["threshold"]
    )
    metrics["accuracy"] = compute_accuracy(pos_scores, neg_scores, metrics["threshold"])
    
    print(f"At threshold {operating_point['threshold']:.3f}")
    print(f"  TPR = {operating_point["TPR"]*100:.1f}%")
    print(f"  FPR = {operating_point["FPR"]*100:.1f}%")
    
    plot_auc_graph(metrics)
    plot_confusion_matrix(pos_scores, neg_scores, metrics["threshold"])
    
    return {
        "embeddings": embeddings.cpu(),
        "labels": labels,
        "pos_scores": sum(pos_scores) / len(pos_scores),
        "neg_scores": sum(neg_scores) / len(neg_scores),
        "metrics": metrics,
    }
    
def embed_all(
    model: nn.Module, 
    dataloader: DataLoader[Tuple[torch.Tensor, str]], 
    device: torch.device
) -> Tuple[torch.Tensor, List[str]]:
    
    model.eval()
    embeddings_list: List[torch.Tensor] = []
    labels_list: List[str] = []
    
    with torch.no_grad():
        for images, _signer_ids in dataloader:
            images = images.to(device)
            embedding = model(images)
            embeddings_list.append(embedding.cpu())
            labels_list.append(_signer_ids)
            
    all_embeddings = torch.cat(embeddings_list)
    
    return all_embeddings, labels_list

def build_verification_pairs(
    embeddings: torch.Tensor,
    test_map: Dict[str, Dict[str, List[str]]]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]: 
    num_samples = embeddings.shape[0]
    
    ordered_paths: List[Tuple[str, str, str]] = []
    
    for signer_id in sorted(test_map.keys(), key=int):
        for path in test_map[signer_id].get("original", []):
            ordered_paths.append((signer_id, "original", path))
    
        for path in test_map[signer_id].get("forged", []):
            ordered_paths.append((signer_id, "forged", path))
    
    assert len(ordered_paths) == num_samples, \
        "Embedding count does not match number of test images."
        
    orig_indices: Dict[str, List[int]] = {}
    forg_indices: Dict[str, List[int]] = {}
    
    for index, (sid, t, _) in enumerate(ordered_paths):
        if t == "original": orig_indices.setdefault(sid, []).append(index)
        else: forg_indices.setdefault(sid, []).append(index)
    
    pos_pairs: List[Tuple[int, int]] = []
    for sid, indices in orig_indices.items():
        if len(indices) >= 2:
            for i, j in combinations(indices, 2):
                pos_pairs.append((i, j))
    
    intra_neg_pairs: List[Tuple[int, int]] = []
    for sid in orig_indices.keys():
        o_idxs = orig_indices[sid]
        f_idxs = forg_indices.get(sid, [])
        for i in o_idxs:
            for j in f_idxs:
                intra_neg_pairs.append((i, j))
    
    inter_neg_pairs: List[Tuple[int, int]] = []
    # original-only to ensure consistency
    sids = list(orig_indices.keys()) 

    for sid_a, sid_b in combinations(sids, 2):
        for i in orig_indices[sid_a]:
            for j in orig_indices[sid_b]:
                inter_neg_pairs.append((i, j))
                
    return pos_pairs, intra_neg_pairs, inter_neg_pairs
    
def compute_scores(
    embeddings: torch.Tensor,
    pairs: List[Tuple[int, int]]
) -> List[float]:
    scores: List[float] = []
    if len(pairs) == 0: return scores
    
    for i, j in pairs:
        if i < 0 or j < 0 or i >= embeddings.shape[0] or j >= embeddings.shape[0]: continue
        
        sim = F.cosine_similarity(
            embeddings[i].unsqueeze(0),
            embeddings[j].unsqueeze(0),
        ).item()
        
        scores.append(float(sim))
    
    return scores    

def compute_metrics(
    pos_scores: List[float],
    neg_scores: List[float]
) -> Dict[str, Any]:
    
    metrics: Dict[str, Any] = {
        "AUC": float("nan"),
        "EER": float("nan"),
        "threshold": float("nan"),
        "fpr": np.array([]),
        "tpr": np.array([]),
    }
    
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return metrics
    
    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_score = pos_scores + neg_scores
    
    try: auc = roc_auc_score(y_true, y_score)
    except Exception:
        metrics["AUC"] = float("nan")
        return metrics
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score) # pyright: ignore[reportUnknownVariableType]

    fnr = 1 - tpr # pyright: ignore[reportUnknownVariableType]
    abs_diffs = np.abs(fnr - fpr) # pyright: ignore[reportUnusedVariable, reportUnknownArgumentType]
    eer_index = int(np.argmin(abs_diffs))
    
    eer = float((fnr[eer_index] + fpr[eer_index]) / 2.0)
    threshold = float(thresholds[eer_index])

    metrics.update({
        "AUC": float(auc),
        "EER": eer,
        "threshold": threshold,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds":thresholds
    })
    
    return metrics

def summarise_operating_point(
    fpr: Any, 
    tpr: Any, 
    thresholds: Any, 
    chosen_threshold: Any
) -> Dict[str, Any]:
    
    index = (np.abs(thresholds - chosen_threshold)).argmin()
    
    return {
        "threshold": thresholds[index],
        "TPR": tpr[index],   # genuine acceptance rate
        "FPR": fpr[index],   # forgery acceptance rate
    }
    
def compute_accuracy(
    pos_scores: List[float], 
    neg_scores: List[float],
    threshold: float
) -> float:
    if len(pos_scores) == 0 and len(neg_scores) == 0: return float("nan")
    
    pos_correct = sum([1 for score in pos_scores if score >= threshold])
    neg_correct = sum([1 for score in neg_scores if score < threshold])
    
    total = len(pos_scores) + len(neg_scores)
    correct = pos_correct + neg_correct
    
    return correct / total if total > 0 else float("nan")

def plot_auc_graph(metrics: Dict[str, Any]) -> None:
    plt.figure() # pyright: ignore[reportUnknownMemberType]
    plt.plot(metrics["fpr"], metrics["tpr"], label=f"AUC = {metrics['AUC']:.3f}") # pyright: ignore[reportUnknownMemberType]
    plt.plot([0,1], [0,1], '--') # pyright: ignore[reportUnknownMemberType]
    plt.xlabel("FPR") # pyright: ignore[reportUnknownMemberType]
    plt.ylabel("TPR") # pyright: ignore[reportUnknownMemberType]
    plt.title("ROC Curve") # pyright: ignore[reportUnknownMemberType]
    plt.legend() # pyright: ignore[reportUnknownMemberType]
    plt.show() # pyright: ignore[reportUnknownMemberType]
    
def plot_confusion_matrix(
    pos_scores: List[float], 
    neg_scores: List[float], 
    threshold: float
) -> None:
    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)

    pos_pred = [1 if score >= threshold else 0 for score in pos_scores]
    neg_pred = [1 if score >= threshold else 0 for score in neg_scores]

    y_pred = pos_pred + neg_pred
        
    cm = confusion_matrix(y_true, y_pred) # type: ignore

    plt.figure(figsize=(8, 6)) # pyright: ignore[reportUnknownMemberType]
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues") # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    plt.title("Confusion Matrix") # pyright: ignore[reportUnknownMemberType]
    plt.ylabel("True Label") # pyright: ignore[reportUnknownMemberType]
    plt.xlabel("Predicted Label") # pyright: ignore[reportUnknownMemberType]
    plt.show() # pyright: ignore[reportUnknownMemberType]