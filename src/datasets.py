import random
from typing import Dict, Tuple, Optional, List, Callable, Iterator
from collections import defaultdict

import torch
import numpy as np
import numpy.typing as npt
# import torchvision.transforms as transforms # pyright: ignore[reportMissingTypeStubs]

from torch.utils.data import Dataset, Sampler
from PIL import Image

class PKSampler(Sampler[List[int]]):
    def __init__(
        self,
        label_to_indices: Dict[str, List[int]],  
        # signers per batch
        P: int,  
        # originals per signer
        K: int,  
        # forgeries per signer (Intra-class)
        F: int,  
        # signatures from other signer (Inter-class)
        M: int,  
        seed: int = 42,
    ):
        # "10_orig", "10_forg" (clear separation of classes)
        self.label_to_indices = label_to_indices
        
        # Signers per batch
        self.P = P
        # Originals per signer
        self.K = K
        # Forgeries mimic the same signer
        self.F = F
        # Signatures from other signers
        self.M = M
        
        self.signers = sorted({lbl.split("_")[0] for lbl in label_to_indices.keys()})
        self.seed = seed

        # Precompute the label pool 
        self._all_indices: List[int] = []
        self._indices_by_label: Dict[str, List[int]] = {}
        self._indices_by_signer: Dict[str, List[int]] = defaultdict(list)
        
        for label, idxs in label_to_indices.items():
            self._indices_by_label[label] = idxs
            signer_id = label.split("_")[0]
            self._indices_by_signer[signer_id].extend(idxs)
            self._all_indices.extend(idxs)

    # The 10 is just heuristic to stretch the epoch length
    def __len__(self) -> int:
        return max(1, len(self.signers) // self.P * 10)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        num_batches = len(self)
        for _ in range(num_batches):
            selected_signers = rng.sample(self.signers, self.P)
            batch: List[int] = []

            for sid in selected_signers:
                # Explicit classification of original and forgeries
                orig_label = f"{sid}_orig"
                forg_label = f"{sid}_forg"

                originals = self._indices_by_label.get(orig_label, [])
                forgeries = self._indices_by_label.get(forg_label, [])

                # (K) Originals (anchor + positives)
                pos = rng.sample(originals, self.K) if len(originals) >= self.K else rng.choices(originals, k=self.K)

                # (F) intra-signer forgeries as hard negatives
                neg_hard = rng.sample(forgeries, self.F) if len(forgeries) >= self.F else rng.choices(forgeries, k=self.F)

                # (M) inter-signer negatives (original and forgeries from other signers)
                global_pool = [idx for idx in self._all_indices if idx not in self._indices_by_signer[sid]]
                neg_global = rng.sample(global_pool, self.M) if len(global_pool) >= self.M else rng.choices(global_pool, k=self.M)

                batch.extend(pos + neg_hard + neg_global)

            yield batch

class SignatureDataset(Dataset[Tuple[torch.Tensor, str]]):
    def __init__(
        self, 
        data_map: Dict[str, Dict[str, List[str]]],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        
        # The data map looks like this: 
        #
        #{
        #    id: {
        #         original: [],
        #         forged: [],
        #     },
        #     id: {
        #         original: [],
        #         forged: [],  
        #     },
        # }
        self.data_map = data_map
        self.transform = transform
        # Get the signer ids in order
        self.signer_ids = sorted(list(data_map.keys()), key=int)
        
        # signer id, image type, index
        self.all_image_references: List[Tuple[str, str, int]] = []
        
        for signer_id in self.signer_ids:
            for index, _ in enumerate(data_map[signer_id].get("original", [])):
                self.all_image_references.append((signer_id, "original", index))
            
            for index, _ in enumerate(data_map[signer_id].get("forged", [])):
                self.all_image_references.append((signer_id, "forged", index))
        
        self.labels = [
            # "orig" or "forg"
            f"{signer_id}_{img_type[:4]}"
            for signer_id, img_type, _ in self.all_image_references
        ]

    def __len__(self)->int:
        return len(self.all_image_references)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        signer_id, image_type, image_index = self.all_image_references[index]
                
        path: str = self.data_map[signer_id][image_type][image_index]
        
        image_pil: Image.Image = Image.open(path).convert("L")
            
        if self.transform is not None:
            image_tensor: torch.Tensor = self.transform(image_pil)
        else:
            image_array: npt.NDArray[np.uint8] = np.array(image_pil, dtype=np.uint8)
            image_tensor: torch.Tensor = torch.from_numpy(image_array).unsqueeze(0).float() / 255.0 # pyright: ignore[reportUnknownMemberType]
        
        label = f"{signer_id}_{image_type[:4]}"
        
        return image_tensor, label

class TestSignatureDataset(Dataset[Tuple[torch.Tensor, str]]):
    def __init__(
        self,
        data_map: Dict[str, Dict[str, List[str]]],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.data_map = data_map
        self.transform = transform

        # Deterministic ordering
        self.signer_ids = sorted(list(data_map.keys()), key=int)

        # Build ordered list matching evaluation logic
        # (signer_id, "original"/"forged", path)
        self.ordered_items: List[Tuple[str, str, str]] = []

        for sid in self.signer_ids:
            for path in data_map[sid].get("original", []):
                self.ordered_items.append((sid, "original", path))

            for path in data_map[sid].get("forged", []):
                self.ordered_items.append((sid, "forged", path))

    def __len__(self) -> int:
        return len(self.ordered_items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        signer_id, _img_type, path = self.ordered_items[index]

        image_pil = Image.open(path).convert("L")

        if self.transform is not None:
            image_tensor: torch.Tensor = self.transform(image_pil)
        else:
            image_array: npt.NDArray[np.uint8] = np.array(image_pil, dtype=np.uint8)
            image_tensor: torch.Tensor = torch.from_numpy(image_array).unsqueeze(0).float() / 255.0 # pyright: ignore[reportUnknownMemberType]

        return image_tensor, signer_id