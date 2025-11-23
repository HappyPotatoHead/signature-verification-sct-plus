from src import DATASET_PATH, LEARNING_CONFIG, OPTIMISER_PARAMS, SCHEDULER_PARAMS, seed_everything, seed_worker
from src import retrieve_signature_images, prepare_signature_map 
from src import training_and_testing_split
from src import TRAIN_TRANSFORM, TEST_TRANSFORM
from src import PKSampler, SignatureDataset, TestSignatureDataset, build_label_to_indices
from src import FeatureExtractionModel, SCTLossWrapper, TripletLoss # pyright: ignore[reportUnusedImport]
from src import Trainer
from src import evaluate

import torch
from pathlib import Path

from torch import Generator
from torch.utils.data import DataLoader

if __name__ == "__main__":
    seed_everything(42)
    g = Generator()
    g.manual_seed(42)

    dataset_path: Path = Path(DATASET_PATH["CEDAR"])
    original_signatures = retrieve_signature_images(dataset_path / "original")
    forged_signatures = retrieve_signature_images(dataset_path / "forged")

    signature_map = prepare_signature_map(original_signatures, forged_signatures)

    train_map, val_map, test_map = training_and_testing_split(
        signature_map, test_ratio=0.2, random_state=42
    )

    train_dataset = SignatureDataset(train_map, TRAIN_TRANSFORM)
    val_dataset = SignatureDataset(val_map, TEST_TRANSFORM)
    test_dataset = TestSignatureDataset(test_map, TEST_TRANSFORM)

    label_to_indices = build_label_to_indices(train_dataset)
    train_sampler = PKSampler(label_to_indices, 8, 2, 2, 2)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_sampler=train_sampler, 
        pin_memory=True, 
        worker_init_fn=seed_worker, 
        generator=g,
        num_workers=4
    )

    val_dataloader = DataLoader(
        val_dataset, 
        int(LEARNING_CONFIG["BATCH_SIZE"]), 
        shuffle=False, 
        pin_memory=True,
        num_workers=4, 
        worker_init_fn=seed_worker
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        int(LEARNING_CONFIG["BATCH_SIZE"]),
        shuffle=False
    )
    
    model = FeatureExtractionModel("efficientnet_v2_m", 256, "IMAGENET1K_V1")
    
    # loss_function = SCTLossWrapper(method="sct", lam=1.0, margin=0.5, verbose = True)
    loss_function = TripletLoss(0.5,"batch_hard", use_diversity=False)
    
    model_trainer = Trainer(
        model,
        loss_function,
        LEARNING_CONFIG,
        OPTIMISER_PARAMS,
        SCHEDULER_PARAMS,
        True
    )
        
    model_trainer.fit(train_dataloader, val_dataloader)
    print(evaluate(model, test_dataloader, test_map, torch.device("cuda")))