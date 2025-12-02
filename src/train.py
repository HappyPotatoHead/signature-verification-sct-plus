from src import SchedulerConfig, ModelState

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as  lr_scheduler
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score  # pyright: ignore[reportUnknownVariableType]

class Trainer:
    def __init__(
        self, 
        model: nn.Module,
        loss_function: nn.Module,
        learning_config: Dict[str, str | int | float],
        optimiser_config: Dict[str, str | float],
        scheduler_config: SchedulerConfig,
        save_checkpoints: bool = True
    ) -> None:
        
        self.model = model
        
        # Training loop
        self.epoch = int(learning_config["EPOCH"])
        self.batch_size = int(learning_config["BATCH_SIZE"])
        self.lr = float(learning_config["LEARNING_RATE"])
        self.early_stop = int(learning_config["EARLY_STOPPING_PATIENT"])
        self.checkpoint_path = Path(str(learning_config["CHECKPOINT_DIR"]))
        self.save_checkpoints = bool(save_checkpoints)
        self.device = torch.device(str(learning_config["DEVICE"]))
        self.device_type = self.device.type
        self.global_step = 0
        
        self.best_val_auc = float("inf")
        self.patience_counter = 0
        
        # Optimiser
        self.optimiser = self._build_optimiser(optimiser_config)
        
        # Scheduler
        self.scheduler = self._build_scheduler(scheduler_config)

        # Loss function
        self.loss_function = loss_function
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Moving to GPU
        self.model.to(self.device)
        
        # checkpointing
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard
        log_dir = learning_config["LOG_DIR"]
        self.writer = SummaryWriter(log_dir)
        
    def train_epoch(
        self, dataloader: DataLoader[Tuple[torch.Tensor, str]]
    ) -> float:
        self.model.train()
        running_loss: float = 0.0
        num_batches: int = len(dataloader)
        
        for batch_index, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            unique_labels = sorted(set(labels))
            label_to_int = {label: i for i, label in enumerate(unique_labels)}
            label_tensor = torch.tensor([
                label_to_int[label] for label in labels
                ], dtype=torch.long, device=self.device)
        
            self.optimiser.zero_grad()
            
            with autocast(device_type=self.device_type):
                outputs = self.model(images)
                
                # SCTLoss returns (loss, triplet_vals, triplet_idxs, hn_ratio, Pos_log, Neg_log)
                loss, triplet_vals, _triplet_idxs, hn_ratio, pos, neg= self.loss_function(outputs, label_tensor)
                
                # Vanilla Triplet Loss returns (loss, stats)
                # loss, stats = self.loss_function(outputs, label_tensor)
            
            # Gradient clipping is optional
            
            self.scaler.scale(loss).backward()  # pyright: ignore[reportUnknownMemberType]
            self.scaler.step(self.optimiser)
            self.scaler.update()
            
            # OneCycleLR steps for batch, not epoch
            if isinstance(self.scheduler, lr_scheduler.OneCycleLR):
                self.scheduler.step()

            running_loss += loss.item()
            self.global_step += 1
            
            if batch_index % 19 == 0:
                # For both SCT and vanilla Triplet Loss
                self.writer.add_scalar(f"Train/BatchLoss", loss.item(), self.global_step) # pyright: ignore[reportUnknownMemberType]
                
                # SCT
                self.writer.add_scalar(f"Train/HN_Ratio", hn_ratio.item(), self.global_step) # pyright: ignore[reportUnknownMemberType]
                self.writer.add_scalar(f"Train/PosMean", pos.mean().item(), self.global_step) # pyright: ignore[reportUnknownMemberType]
                self.writer.add_scalar(f"Train/NegMean", neg.mean().item(), self.global_step) # pyright: ignore[reportUnknownMemberType]
                
                self.writer.add_histogram(f"Train/Pos", triplet_vals[:,0], self.global_step) # pyright: ignore[reportUnknownMemberType]
                self.writer.add_histogram(f"Train/Neg", triplet_vals[:,1], self.global_step) # pyright: ignore[reportUnknownMemberType]
                
                # Vanilla Triplet Loss
                # self.writer.add_scalar("Train/HN_Ratio", stats["hn_ratio"], self.global_step) # pyright: ignore[reportUnknownMemberType]
                # self.writer.add_scalar("Train/PosMean", stats["pos"].mean().item(), self.global_step) # pyright: ignore[reportUnknownMemberType]
                # self.writer.add_scalar("Train/NegMean", stats["neg"].mean().item(), self.global_step) # pyright: ignore[reportUnknownMemberType]

                # self.writer.add_histogram("Train/Pos", stats["triplet_vals"][:,0], self.global_step) # pyright: ignore[reportUnknownMemberType]
                # self.writer.add_histogram("Train/Neg", stats["triplet_vals"][:,1], self.global_step) # pyright: ignore[reportUnknownMemberType]
                
                total_norm: float = 0.
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += float(p.grad.data.norm(2).item()) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                        
                self.writer.add_scalar("Gradients/TotalNorm", total_norm, self.global_step) # pyright: ignore[reportUnknownMemberType]
        
        return running_loss / num_batches
        
    def evaluate(
        self,
        dataloader: DataLoader[Tuple[torch.Tensor, str]] 
    ) -> Tuple[
            Dict[str, float], 
            torch.Tensor, 
            List[str]
    ]:
        self.model.eval()
        all_embeddings_list: List[torch.Tensor] = []
        all_labels: List[str] = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                
                with autocast(device_type=self.device_type):
                    embeddings = self.model(images)
                    all_embeddings_list.append(embeddings.cpu())
                    all_labels.extend(labels)
        
        all_embeddings = torch.cat(all_embeddings_list, dim=0)
        n = len(all_labels)
        
        similarity_matrix = F.cosine_similarity(
            all_embeddings.unsqueeze(1),
            all_embeddings.unsqueeze(0),
            dim=2
        )
        
        signer_ids = [label.split("_")[0] for label in all_labels]
        intra_mask = torch.zeros((n, n), dtype=torch.bool)
        inter_mask = torch.zeros((n, n), dtype=torch.bool)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if signer_ids[i] == signer_ids[j]:
                    intra_mask[i, j] = True
                else:
                    inter_mask[i, j] = True

        intra_sims: torch.Tensor = similarity_matrix[intra_mask] 
        inter_sims: torch.Tensor = similarity_matrix[inter_mask]  

        intra_sims_np = intra_sims.cpu().numpy() # type: ignore
        inter_sims_np = inter_sims.cpu().numpy() # type: ignore

        # Metrics
        y_true = [1] * len(intra_sims_np) + [0] * len(inter_sims_np) # type: ignore
        y_scores = list(intra_sims_np) + list(inter_sims_np) # type: ignore
        auc = roc_auc_score(y_true, y_scores)

        metrics: Dict[str, float | int] = {
            "AUC": auc, # type: ignore
            "mean_intra_similarity": float(intra_sims.mean()),
            "mean_inter_similarity": float(inter_sims.mean()),
            "num_intra_pairs": len(intra_sims),
            "num_inter_pairs": len(inter_sims)
        }

        return metrics, all_embeddings, all_labels
    
    def fit(
        self, 
        train_dataloader: DataLoader[Tuple[torch.Tensor, str]],
        val_dataloader: DataLoader[Tuple[torch.Tensor, str]]
    ) -> None: 
        for epoch in range(self.epoch):
            train_loss = self.train_epoch(train_dataloader)
            val_metrics, val_embedding, all_labels = self.evaluate(val_dataloader)
            
            self.writer.add_embedding( # pyright: ignore[reportUnknownMemberType]
                mat = val_embedding,
                metadata = all_labels,
                global_step = self.global_step
            )
            
            self.writer.add_scalar("Loss/train", train_loss, epoch) # pyright: ignore[reportUnknownMemberType]
            self.writer.add_scalar("AUC/val", val_metrics["AUC"], epoch) # pyright: ignore[reportUnknownMemberType]
            self.writer.add_scalar("Loss/1_minus_auc", 1.0 - val_metrics["AUC"], epoch) # pyright: ignore[reportUnknownMemberType] 
            self.writer.add_scalar("Learning rate", self.optimiser.param_groups[0]["lr"], epoch) # pyright: ignore[reportUnknownMemberType]
            
            if self.scheduler is not None and not isinstance(self.scheduler, lr_scheduler.OneCycleLR):
                self.scheduler.step()

            print(f"Epoch [{epoch + 1}/{self.epoch}]"
                  f"| Train loss: {train_loss:.4f}"
                  f"| AUC: {val_metrics["AUC"]:.4f}")
        
            val_auc_for_stop = 1.0 - val_metrics["AUC"]
            if val_auc_for_stop < self.best_val_auc:
                self.best_val_auc = val_auc_for_stop
                self.patience_counter = 0
                if self.save_checkpoints:
                    self._save_checkpoint(epoch, val_auc_for_stop, self.best_val_auc, self.patience_counter)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop:
                    print("Early stopping")
                    break
        
        self.writer.close()
    
    def load_checkpoint(
        self, 
        path: str
    ) -> None:
        
        self._has_path(path)
        try:
            checkpoint_model: ModelState = torch.load(path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint from {path}: {e}")
        
        self.model.load_state_dict(checkpoint_model["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint_model["optimiser_state_dict"])
        
        if self.scheduler and checkpoint_model["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint_model["scheduler_state_dict"])
        
        self.epoch = checkpoint_model["epoch"]
        self.best_val_auc = checkpoint_model["best_loss"]
        self.patience_counter = checkpoint_model["patience_counter"]
    
    def _save_checkpoint(
        self, 
        epoch: int, 
        auc: float, 
        best_auc: float, 
        patience_counter: int,
    ) -> None:
        model_state: ModelState = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "auc": auc,
            "best_loss": best_auc,
            "patience_counter": patience_counter
        }
        
        torch.save(
            model_state, self.checkpoint_path / f"{epoch+1}_auc_{auc:.4f}.pt"
        )
    
    def _has_path(self, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found at {path}")
        print(f"File is ok!")
                
    def _build_optimiser(self, optimiser_config: Dict[str, str | float]) -> optim.Optimizer:
        optimiser_name =  str(optimiser_config["optimiser"])
        optimiser_class = getattr(optim, optimiser_name)
        
        optimiser_params = {**optimiser_config}
        optimiser_params.pop("optimiser")
        optimiser_params["lr"] = self.lr
        
        return optimiser_class(self.model.parameters(), **optimiser_params)
        
    def _build_scheduler(
        self, 
        scheduler_config: SchedulerConfig,
        ) -> Optional[lr_scheduler.LRScheduler]:
        schedulers: List[lr_scheduler.LRScheduler] = []
         
        for sched_cfg in scheduler_config.get("SCHEDULERS", []):
            name = sched_cfg["name"]
            params = sched_cfg.get("params", {})
            sched_class = getattr(lr_scheduler, name)
            schedulers.append(sched_class(self.optimiser, **params))

        if scheduler_config.get("SCHEDULER") == "SequentialLR":
            return lr_scheduler.SequentialLR(
                self.optimiser,
                schedulers=schedulers,
                milestones=scheduler_config.get("MILESTONES", [])
            )
        
        return schedulers[0] if schedulers else None