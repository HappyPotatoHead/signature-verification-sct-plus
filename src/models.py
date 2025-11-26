from src import BACKBONE_CONFIG

import sys
from typing import Any, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model_weights # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]

class FeatureExtractionModel(nn.Module):
    def __init__(
        self,
        backbone_type: str,
        embedding_dim: int = 256,
        weights: Optional[str] = None, 
        dropout_rate: float  = 0.4,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self._check_parameters(backbone_type, embedding_dim, dropout_rate)
        
        FREEZE_UP_TO: int = 4
        
        self.embedding_dim = embedding_dim
        self.backbone_type = backbone_type
        self.dropout_rate = dropout_rate  
        
        self.weights = weights
        self.weights_enum = None
        
        backbone_builder = BACKBONE_CONFIG[self.backbone_type]["builder"]
        backbone_out_channels = BACKBONE_CONFIG[self.backbone_type]["out_channels"]
        
        self._retrieve_weights(str(self.weights), backbone_builder)
        
        self.model = backbone_builder(weights = self.weights_enum)
        
        first_layer = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels= 1,
            out_channels=first_layer.out_channels,
            kernel_size=first_layer.kernel_size,
            stride=first_layer.stride,
            padding=first_layer.padding,
            bias=first_layer.bias is not None,
        )
        
        if self.weights is not None:
            with torch.no_grad():
                self.model.features[0][0].weight.data = (
                    first_layer.weight.data.mean(dim=1, keepdim=True)
                )
        else:
            nn.init.kaiming_normal_(
                self.model.features[0][0].weight, mode="fan_out", nonlinearity="relu"
            )
            if self.model.features[0][0].bias is not None:
                nn.init.constant_(self.model.features[0][0].bias, 0)

        self.backbone = self.model.features

        # Zero shot Learning       
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        for i, block in enumerate(self.model.features):
            if i <= FREEZE_UP_TO:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True
            
        self.pool1 = nn.AdaptiveAvgPool2d(1) 
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(backbone_out_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, self.embedding_dim)
        self.bn2 = nn.BatchNorm1d(self.embedding_dim) 
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        
        x = self.pool1(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        # x = self.relu(x)
        
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def _check_parameters(
        self,
        backbone_type: str,
        embedding_dim: int,
        dropout_rate: float,
    ) -> None:
        if embedding_dim <= 0:
            raise ValueError(
                f"Embedding dimension must be positive, got {embedding_dim}"
            )
        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError(
                f"Dropout rate must be between 0 and 1, got {dropout_rate}"
            )
        if backbone_type not in BACKBONE_CONFIG:
            raise ValueError(
                f"Unsupported backbone type: {backbone_type}. Choose from {list(BACKBONE_CONFIG.keys())}"
            )
    
    def _retrieve_weights(
        self, 
        weights: str, 
        backbone_builder: Any
    ) -> None:
        try:
            weights_enum_type = get_model_weights(backbone_builder)
            if hasattr(weights_enum_type, weights):
                self.weights_enum = getattr(weights_enum_type, weights)
            else:
                print(
                    f"Warning: Could not get weights type for backbone '{self.backbone_type}'. Using default random initialisation.",
                    file=sys.stderr,
                )
                weights = ""
        except AttributeError:
            print(
                f"Warning: Specified weights alias '{weights}' not found for {self.backbone_type}. Check available weights in torchvision documentation. Using default random initialisation.",
                file=sys.stderr,
            )
            weights = ""
        except Exception as e:
            print(
                f"Warning: An unexpected error occurred looking up weights '{weights}' for {self.backbone_type}: {e}. Using default random initialisation.",
                file=sys.stderr,
            )
            
            weights = ""
 
class TripletLoss(nn.Module):
    def __init__(
        self, 
        margin: float = 1.0,
        mining_strategy: str = "batch_semi_hard",
        use_diversity: bool = False,
        lambda_diversity: float = 0.1,
        p: int = 2,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        
        if margin < 0:
            raise ValueError(f"Margin msut be non-negative, got {margin}")
        if mining_strategy not in ["batch_hard", "batch_semi_hard"]:
            raise ValueError(f"Invalid mining strategy, got {mining_strategy}")
        
        self.margin = margin
        self.mining_strategy = mining_strategy
        
        # This is optional, but it's usually not needed 
        self.lambda_diversity = lambda_diversity
        self.use_diversity = use_diversity
        
        self.p = p
    
        # Calculate margin automatically
        self.base_loss = nn.TripletMarginLoss(margin=margin, p=p)
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=self.p)
        mask_anchor_positive, mask_anchor_negative = self._get_triplet_mask(labels)
        
        if mask_anchor_positive.sum().item() == 0:
            print("No positive pairs in batch")
        
        if mask_anchor_negative.sum().item() == 0:
            print("No negative pairs in batch")

        if self.mining_strategy == "batch_hard":
            pos_idx, neg_idx = self._batch_hard_mining(distances, mask_anchor_positive, mask_anchor_negative)
        else:
            pos_idx, neg_idx, has_semi = self._batch_semi_hard_mining(distances, mask_anchor_positive, mask_anchor_negative)
        
            if not has_semi.any(): 
                print("No semi-hard negatives found (fallback to hard negatives)")

        # anchors = embeddings
        positives = embeddings[pos_idx]
        negatives = embeddings[neg_idx]
        
        batch_indices = torch.arange(distances.size(0), device=distances.device)
        pos_dist = distances[batch_indices, pos_idx]
        neg_dist = distances[batch_indices, neg_idx]
        
        triplet_vals = torch.stack([pos_dist, neg_dist], dim = 1)
        
        hn_ratio = (neg_dist < pos_dist).float().mean() if pos_dist.numel() else torch.tensor(0.0)
        
        stats: Dict[str, Any] = {
            "pos": pos_dist.detach(), 
            "neg": neg_dist.detach(),
            "triplet_vals": triplet_vals.detach(),
            "hn_ratio": hn_ratio.item(),
            "pos_idx": pos_idx.detach(),
            "neg_idx": neg_idx.detach(),
        }
        
        triplet_loss = self.base_loss(embeddings, positives, negatives)

        if self.use_diversity and self.lambda_diversity > 0:
            triplet_loss = triplet_loss + self._compute_diversity_regularisation(embeddings)

        return triplet_loss, stats

    # Gives me 1s and 0s
    def _get_triplet_mask(
        self, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = labels.unsqueeze(1)
        mask_anchor_positive = (labels == labels.T) & ~torch.eye(
            labels.size(0), dtype=torch.bool, device=labels.device
        )
        mask_anchor_negative = labels != labels.T
        
        return (mask_anchor_positive, mask_anchor_negative)
    
    def _batch_hard_mining(
        self,
        distances: torch.Tensor,
        mask_anchor_positive: torch.Tensor,
        mask_anchor_negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        positive_distance = torch.where(
            mask_anchor_positive, distances, torch.full_like(distances, -float("inf"))
        )
        
        _max_positive_distance, pos_idx = torch.max(positive_distance, dim=1)
        
        negative_distance = torch.where(
            mask_anchor_negative, distances, torch.full_like(distances, float("inf"))
        )
        _min_negative_distance, neg_idx = torch.min(negative_distance, dim=1)
        
        return pos_idx, neg_idx

    def _batch_semi_hard_mining(
        self,
        distances: torch.Tensor,
        mask_anchor_positive: torch.Tensor,
        mask_anchor_negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positive_distance_filtered = torch.where(
            mask_anchor_positive, distances, torch.full_like(distances, -float("inf"))
        )
        hardest_positive_dist, pos_idx = torch.max(positive_distance_filtered, dim=1)
        
        valid_negative_distances = torch.where(
            mask_anchor_negative, distances, torch.full_like(distances, float("inf"))
        )
        
        is_harder_than_positive = valid_negative_distances > hardest_positive_dist.unsqueeze(1)
        
        is_easier_than_margin = valid_negative_distances < (
            hardest_positive_dist.unsqueeze(1) + self.margin
        )
        
        semi_hard_mask = mask_anchor_negative & is_harder_than_positive & is_easier_than_margin

        semi_hard_negatives = torch.where(
            semi_hard_mask, distances, torch.full_like(distances, float("inf"))
        )
        
        _easiest_semi_hard, semi_idx = torch.min(semi_hard_negatives, dim=1)
        _hardest_overall, hard_idx = torch.min(valid_negative_distances, dim=1)

        use_semi = semi_hard_mask.sum(dim=1) > 0
        neg_idx = torch.where(use_semi, semi_idx, hard_idx)
        
        return pos_idx, neg_idx, use_semi
    
    # This is not really necessary
    def _compute_diversity_regularisation(self, embeddings: torch.Tensor) -> torch.Tensor:
        sim = embeddings @ embeddings.T
        eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
        diversity = (sim - eye).pow(2).mean()
        return self.lambda_diversity * diversity
 
class SCTLossWrapper(nn.Module):
    def __init__(
        self, 
        method: str = "sct", 
        lam: float = 1.0,
        margin: float = 1.0,
        verbose: bool = False
    ) -> None:
        super().__init__() # pyright: ignore[reportUnknownMemberType]
        self.loss_fn = SCTLoss(method, lam, margin, verbose)

    def forward(
        self, 
        fvec: torch.Tensor, 
        Lvec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # SCTLoss returns (loss, Triplet_val.clone().detach().cpu(), Triplet_idx.clone().detach().cpu(), hn_ratio, Pos, Neg)
        return self.loss_fn(fvec, Lvec)

class SCTLoss(nn.Module):
    def __init__(
        self, 
        method: str, 
        lam: float=1.0, 
        margin: float = 1.0,
        verbose: bool = False,
    ):
        super(SCTLoss, self).__init__() # pyright: ignore[reportUnknownMemberType]
        
        if method == 'sct':
            self.sct, self.semi = True, False
        elif method == 'hn':
            self.sct, self.semi = False, False
        elif method == 'shn':
            self.sct, self.semi = False, True
        else: raise ValueError('loss type is not supported')
        self.lam = lam
        self.margin = margin
        self.verbose = verbose

    def forward(
        self,
        fvec: torch.Tensor,
        Lvec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        device = fvec.device
        Same, Diff = self._build_boolean_masks(Lvec.view(-1))
        CosSim = self._calculate_cosine_similarity(fvec, fvec)
        
        Pos, I_pos, Mask_pos_valid, _Pos_log = self._select_positives(CosSim, Diff)
        Neg, I_neg, Mask_neg_valid, _Neg_log = self._select_negatives(CosSim, Same, Diff, Pos)

        Triplet_val, Triplet_idx, HardMask, EasyMask, hn_ratio = self._build_triplets(
            Pos, Neg, I_pos, I_neg, Mask_pos_valid, Mask_neg_valid
        )
        
        loss = self._compute_loss(Triplet_val, Pos, Neg, HardMask, EasyMask, device)

        if self.verbose:
            hr = float(hn_ratio) if isinstance(hn_ratio, (float, int)) or (hasattr(hn_ratio, "item") and not torch.isnan(hn_ratio)) else 0.0
            print(f'loss:{loss.item():.4f} hn_rt:{hr:.4f}', end='\r')
            
        return (
            loss, 
            Triplet_val.clone().detach().cpu(), 
            Triplet_idx.clone().detach().cpu(), 
            hn_ratio if isinstance(hn_ratio, float) else hn_ratio.clone().detach().cpu(),
            Pos.detach().cpu(), 
            Neg.detach().cpu()
        )
        
    def _select_positives(
        self, 
        CosSim: torch.Tensor, 
        Diff: torch.Tensor
    ) -> Tuple[
            torch.Tensor, 
            torch.Tensor, 
            torch.Tensor,
            torch.Tensor,
        ]:
        
        D_pos = CosSim.clone().detach()
        D_pos[Diff] = -1
        # D_pos[D_pos>0.9999] = -1
        V_pos, I_pos = D_pos.max(1)
 
        Mask_pos_valid = (V_pos > -1) & (V_pos < 1)
        Pos = CosSim[torch.arange(0, CosSim.size(0)), I_pos]
        Pos_log = Pos.clone().detach().cpu()
    
        return Pos, I_pos, Mask_pos_valid, Pos_log
    
    def _select_negatives(
        self, 
        CosSim: torch.Tensor,
        Same: torch.Tensor,
        Diff: torch.Tensor, 
        V_pos: torch.Tensor
    ) -> Tuple[
            torch.Tensor, 
            torch.Tensor, 
            torch.Tensor,
            torch.Tensor,
        ]:
        
        D_neg = CosSim.clone().detach()
        D_neg[Same] = -1
        
        # Masking out non-Semi-Hard Negative
        if self.semi:    
            D_neg[(D_neg > V_pos.unsqueeze(1)) & Diff] = -1 
            # D_neg[(D_neg > (V_pos.repeat(CosSim.size(0), 1).t())) & Diff] = -1
            
        V_neg, I_neg = D_neg.max(1)
        Mask_neg_valid = (V_neg > -1) & (V_neg < 1)
        Neg = CosSim[torch.arange(0, CosSim.size(0)), I_neg]
        Neg_log = Neg.clone().detach().cpu()
        
        return Neg, I_neg, Mask_neg_valid, Neg_log 
        
    def _build_triplets(
        self, 
        Pos: torch.Tensor, 
        Neg: torch.Tensor,
        I_pos: torch.Tensor,
        I_neg: torch.Tensor,
        Mask_pos_valid: torch.Tensor,
        Mask_neg_valid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        Mask_valid = Mask_pos_valid & Mask_neg_valid
        HardMask = ((Neg > Pos) | (Neg > 0.8)) & Mask_valid
        EasyMask = ((Neg < Pos) & (Neg < 0.8)) & Mask_valid
        hn_ratio = (Neg>Pos)[Mask_valid].clone().float().mean().cpu()
        
        Triplet_val = torch.stack([Pos, Neg], 1)
        Triplet_idx = torch.stack([I_pos, I_neg], 1)
        
        return Triplet_val, Triplet_idx, HardMask, EasyMask, hn_ratio
    
        # Triplet_val_log = Triplet_val.clone().detach().cpu()
        # Triplet_idx_log = Triplet_idx.clone().detach().cpu()
        
    def _compute_loss(
        self, 
        Triplet_val: torch.Tensor, 
        Pos: torch.Tensor, 
        Neg: torch.Tensor,
        HardMask: torch.Tensor, 
        EasyMask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if self.sct:    
            loss_hard = Neg[HardMask].sum()
            N_hard = HardMask.float().sum().item()
            if torch.isnan(loss_hard) or N_hard == 0:
                loss_hard, N_hard = torch.tensor(0.0), 0
                print('No hard triplets in the batch')
                
            loss_easy = -F.log_softmax(
                Triplet_val[EasyMask, :] / 0.1, dim=1
            )[:, 0].sum()
            N_easy = EasyMask.float().sum().item()
            if torch.isnan(loss_easy) or N_easy == 0:
                loss_easy, N_easy = torch.tensor(0.0), 0
                print('No easy triplets in the batch')
            
            pos_valid = (Pos > -1) & ( Pos < 1)
            N_pos = int(pos_valid.float().sum().item())
            if N_pos > 0:
                positive_pull = F.relu(self.margin - Pos[pos_valid]).mean()
            else:
                positive_pull = torch.Tensor(0.0, device=device)
            
            N_total = max(N_hard + N_easy, 1)
            sct_loss = (loss_easy + self.lam * loss_hard) / N_total
            return sct_loss + 0.5 * positive_pull
            # return (loss_easy + self.lam * loss_hard + 0.5 * positive_pull) / N_total
            # return (loss_easy + self.lam * loss_hard) / N_total
        else:
            return -F.log_softmax(Triplet_val / 0.1, dim=1)[:, 0].mean()

            # return -F.log_softmax(
            #     Triplet_val[Triplet_val, :] / 0.1, dim=1
            # )[:, 0].mean()

    def _calculate_cosine_similarity(
        self,     
        Mat_A: torch.Tensor, 
        Mat_B: torch.Tensor, 
        norm:int=1, 
    ) -> torch.Tensor: 
        
        Mat_A = F.normalize(Mat_A, p=2, dim=1)
        Mat_B = F.normalize(Mat_B, p=2, dim=1)
        # _N_A = Mat_A.size(0)
        # _N_B = Mat_B.size(0)
    
        D = Mat_A.mm(torch.t(Mat_B))
    
        # Ignore self-similarity
        D.fill_diagonal_(-norm)
        return D
    
    def _build_boolean_masks(
        self, 
        Lvec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # N = Lvec.size(0)
        # Forms N x N
        # Mask = Lvec.repeat(N,1)
        
        # True if labels match
        Same = Lvec.unsqueeze(0) == Lvec.unsqueeze(1)
        # Same = (Mask == Mask.t())
        
        # Same / Different masks
        return Same.clone().fill_diagonal_(0), ~Same