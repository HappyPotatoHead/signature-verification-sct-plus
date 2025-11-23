from typing import Dict, Tuple, Any, List
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve # pyright: ignore[reportUnknownVariableType]

def evaluate(
    model: nn.Module, 
    test_loader: DataLoader[Tuple[torch.Tensor, str]], 
    test_map:  Dict[str, Dict[str, List[str]]],
    device: torch.device
) -> Dict[str, Any]:
    model.to(device)
    
    # Applying the model
    embeddings = embed_all(model, test_loader, device)

    # original, easy, and hard forgery  
    pos_pairs, intra_pairs, inter_pairs = build_verification_pairs(embeddings, test_map)

    # Intra and Inter
    neg_pairs = intra_pairs + inter_pairs
    
    # Compute similarities
    pos_scores = compute_scores(embeddings, pos_pairs)
    neg_scores = compute_scores(embeddings, neg_pairs)

    # Metrics
    metrics = compute_metrics(pos_scores, neg_scores)

    return {
        "embeddings": embeddings,
        "pos_scores": pos_scores,
        "neg_scores": neg_scores,
        "metrics": metrics,
    }

def embed_all(
    model: nn.Module, 
    dataloader: DataLoader[Tuple[torch.Tensor, str]], 
    device: torch.device
) -> torch.Tensor:
    
    model.eval()
    embeddings_list: List[torch.Tensor] = []
    # labels_list: List[str] = []
    
    with torch.no_grad():
        for images, _signer_ids in dataloader:
            images = images.to(device)
            embedding = model(images)
            embeddings_list.append(embedding.cpu())
            # labels_list.append(signer_ids.cpu())
            
    all_embeddings = torch.cat(embeddings_list)
    # labels = torch.cat(labels_list)
    
    return all_embeddings
# , labels

def build_verification_pairs(
    embeddings: torch.Tensor,
    test_map: Dict[str, Dict[str, List[str]]]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]: 
    num_samples = embeddings.shape[0]
    
    ordered_paths: List[Tuple[str, str, str]] = []
    # ordered_types = []
    
    for signer_id in sorted(test_map.keys(), key=int):
        for path in test_map[signer_id].get("original", []):
            ordered_paths.append((signer_id, "original", path))
    
        for path in test_map[signer_id].get("forged", []):
            ordered_paths.append((signer_id, "forged", path))
    
    assert len(ordered_paths) == num_samples, \
        "Embedding count does not match number of test images."
        
    # signer_to_indices = {}
    orig_indices: Dict[str, List[int]] = {}
    forg_indices: Dict[str, List[int]] = {}
    
    for index, (sid, t, _) in enumerate(ordered_paths):
        # signer_to_indices.setdefault(sid, []).append(index)
        if t == "original":
            orig_indices.setdefault(sid, []).append(index)
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
    sids = list(orig_indices.keys())  # original-only ensures consistency

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
    if len(pairs) == 0:
        return scores
    
    for i, j in pairs:
        if i < 0 or j < 0 or i >= embeddings.shape[0] or j >= embeddings.shape[0]:
            continue
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
    
    try:
        auc = roc_auc_score(y_true, y_score)
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
    })
    
    return metrics

# def evaluate_model(
#     model: nn.Module,
#     test_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
#     margin: float = 0.5,
# ) -> EvalResults:
#     model.eval()
#     all_labels: List[float] = []
#     all_distances: List[float] = []
#     embeddings_list: defaultdict[
#         str, 
#         List[
#             npt.NDArray[
#                 np.float32
#             ]
#         ]
#     ] = defaultdict(list)
    
#     total_loss: float = 0.0
#     num_batches: int = 0
    
#     with torch.no_grad():
#         for anchor, positive, negative, _anchor_id in test_loader:
#             anchor = anchor.to(LEARNING_CONFIG["DEVICE"])
#             positive = positive.to(LEARNING_CONFIG["DEVICE"])
#             negative = negative.to(LEARNING_CONFIG["DEVICE"])
             
#             anchor_embedding = model(anchor)
#             positive_embedding = model(positive)
#             negative_embedding = model(negative)
            
#             positive_dist, negative_dist, loss = compute_distance_eval(anchor_embedding, positive_embedding, negative_embedding, margin)
#             total_loss += loss.item()
#             num_batches += 1
            
#             for k, v in zip(("anchor", "positive", "negative"), (anchor_embedding, positive_embedding, negative_embedding)):
#                 embeddings_list[k].append(v.cpu().numpy())
            
#             all_distances.extend(positive_dist.cpu().tolist()); all_labels.extend([1]*len(positive_dist)) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
#             all_distances.extend(negative_dist.cpu().tolist()); all_labels.extend([0]*len(negative_dist)) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]

#     final_embeddings: Dict[str, npt.NDArray[np.float32]] = {k: np.concatenate(v, axis=0) for k,v in embeddings_list.items()}
#     all_distances_np = np.array(all_distances, np.float32).reshape(-1)
#     all_labels_np = np.array(all_labels, np.int64).reshape(-1)
    
#     fpr, tpr, fnr, thresholds, roc_auc, scores = calculate_auc(all_labels_np, all_distances_np)
#     eer, eer_threshold = calculate_eer_eer_threshold(thresholds, fpr, fnr)

#     preds = (scores >= eer_threshold).astype(int)

#     metrics: Dict[str, Any] = {
#         "avg_triplet_loss": total_loss/num_batches,
#         "avg_pos_dist": float(all_distances_np[all_labels_np==1].mean()),
#         "avg_neg_dist": float(all_distances_np[all_labels_np==0].mean()),
#         "auc_roc": float(roc_auc),
#         "eer": float(eer),
#         "eer_threshold": float(eer_threshold),
#         "accuracy": accuracy_score(all_labels_np, preds),
#         "precision": precision_score(all_labels_np, preds, pos_label=1),
#         "recall": recall_score(all_labels_np, preds, pos_label=1),
#     }

#     curves = {"fpr": fpr, "tpr": tpr, "fnr": fnr}
#     return EvalResults(all_distances_np, all_labels_np, final_embeddings, metrics, curves)

# def compute_distance_eval(
#     anchor: torch.Tensor, 
#     positive: torch.Tensor, 
#     negative: torch.Tensor, 
#     margin: float,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     positive_dist = F.pairwise_distance(anchor, positive)
#     negative_dist = F.pairwise_distance(anchor, negative)
#     loss = torch.mean(torch.relu(positive_dist - negative_dist + margin))
#     return positive_dist, negative_dist, loss

# def calculate_eer_eer_threshold(
#     thresholds: npt.NDArray[np.float32],
#     fpr: npt.NDArray[np.float32],
#     fnr: npt.NDArray[np.float32],
# ) -> Tuple[float, float]:
#     absolute_difference = np.abs(fpr - fnr)
#     index = np.argmin(absolute_difference)
#     eer = (fpr[index] + fnr[index]) / 2.0
#     eer_threshold = thresholds[index]
    
#     return eer, eer_threshold

# def calculate_auc(
#     all_labels_np: npt.NDArray[np.int64], 
#     all_distances_np: npt.NDArray[np.float32]
# ) -> Tuple[
#     npt.NDArray[np.float32],
#     npt.NDArray[np.float32],
#     npt.NDArray[np.float32],
#     npt.NDArray[np.float32],
#     float,
#     npt.NDArray[np.float32],
# ]:    
#     scores = -all_distances_np
#     fpr, tpr, thresholds = roc_curve(all_labels_np, scores) # pyright: ignore[reportUnknownVariableType]
#     roc_auc = float(auc(fpr, tpr)) # pyright: ignore[reportUnknownArgumentType]

#     fnr = 1 - tpr  # pyright: ignore[reportUnknownVariableType]

#     return fpr, tpr, fnr, thresholds, roc_auc, scores # pyright: ignore[reportUnknownVariableType]