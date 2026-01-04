# Offline Signature Verification
> This is the final year project for my Bachelor's of Computer Science degree

This model utilises the [EfficientNetV2](https://arxiv.org/abs/2104.00298) as the feature extraction backbone. This is complemented with a custom loss function, $L_{SC+}$, inspired from [Hard negative examples are hard, but useful](https://arxiv.org/abs/2007.12749). This project also feature an extension of $PK$ sampling, $PKFM$, which provides more control over the count of intra-class and inter-class signatures.

![header](assets/header.png)

## Built With

[![Python][python-image]][python-url] `3.13.9`

| Libraries                                            |  Version      |  
|------------------------------------------------------|---------------|
| [![PyTorch][pytorch-image]][pytorch-url]             | `2.9.0+cu130` |
| [![NumPy][numpy-image]][numpy-url]                   | `2.2.6`       |
| [![TensorBoard][tensorboard-image]][tensorboard-url] | `2.20.0`      |
| [![OpenCV][opencv-image]][opencv-url]                | `4.12.0.88`   |
| [![PIL][pillow-image]][pillow-url]                   | `11.3.0`      |
| [![Scikit Learn][sklearn-image]][sklearn-url]        | `1.7.2`       |
| [![Matplotlib][matplotlib-image]][matplotlib-url]    | `3.10.7`      |
| [![Seaborn][seaborn-image]][seaborn-url]             | `0.13.2`      |


## Installation

```bash
git clone https://github.com/HappyPotatoHead/signature-verification-sct-plus
```

## Usage example

### Configurations

Edit these settings to modify the training pipeline

#### Signature images

```python
# Place the online source here
DATASET_SOURCE: Dict[str, str] = {
    "cedar": "shreelakshmigp/cedardataset", 
}

# Raw dataset folder
DATASET_PATH: Dict[str, str] = {
    "CEDAR": "data\\CEDAR"
}

IMAGE_FORMATS: List[str] = [".png", ".jpg", ".jpeg", ".bmp"]
```

```python
# Data augmentations 
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
```

#### Training 

```python
# Training Loop
LEARNING_CONFIG: Dict[str, str | int | float] = {
    "BATCH_SIZE": 32,
    "EPOCH": 50,
    "LEARNING_RATE": 1e-3,
    "EARLY_STOPPING_PATIENT": 10,
    
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    "CHECKPOINT_DIR": "checkpoint/exp_01_sct",
    "LOG_DIR": "runs/exp_01_sct"
}

# Optimer algorithm
OPTIMISER_PARAMS: Dict[str, str | float] = {
    "optimiser": "AdamW",
    # Prevent overwriting the pretrained weights too aggressively
    "weight_decay": 1e-3,
}

# Scheduler algorithm
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
```

### Training The Model

Make changes to these code to change the backbone and the loss function used.

```python
model = FeatureExtractionModel("efficientnet_v2_m", 256, "IMAGENET1K_V1")

loss_function = SCTLossWrapper(
    method = "sct", 
    lam = 1.0, 
    margin = 0.5, 
    positive_pull_weight = 0.5, 
    verbose = True
)

# loss_function = TripletLoss(0.5,"batch_hard")
```

## Development setup

1. Navigate to the project directory
2. Create a virtual environment
3. Run
    ```bash
    pip install -r requirements.txt
    ```
4. Navigate to `main.py` to start training and testing the model. 


## Release History

* 1.0
    * First release! 

## Meta

Jimmy Ding – [Potato Garden](https://potatogarden.surge.sh/) – jimmydingjk@gmail.com

Distributed under the Apache license. See ``LICENSE`` for more information.

[HappyPotatoHead](https://github.com/HappyPotatoHead/)

## Contributing

1. Fork it (<https://github.com/HappyPotatoHead/signature-verification-sct-plus/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->

[python-image]: https://img.shields.io/badge/Python-fffcf0?logo=python&logoColor=100f0f
[python-url]: https://docs.python.org/3/

[pytorch-image]: https://img.shields.io/badge/PyTorch-fffcf0?logo=pytorch&logoColor=100f0f
[pytorch-url]: https://pytorch.org/get-started/locally/

[numpy-image]: https://img.shields.io/badge/NumPy-fffcf0?logo=numpy&logoColor=100f0f
[numpy-url]: https://numpy.org/doc/stable/

[matplotlib-image]: https://img.shields.io/badge/Matplotlib-fffcf0?logo=matplotlib&logoColor=100f0f
[matplotlib-url]: https://matplotlib.org/stable/index.html

[seaborn-image]: https://img.shields.io/badge/Seaborn-fffcf0?logo=matplotlib&logoColor=100f0f
[seaborn-url]: https://seaborn.pydata.org/

[tensorboard-image]: https://img.shields.io/badge/TensorBoard-fffcf0?logo=TensorFlow&logoColor=100f0f
[tensorboard-url]: https://www.tensorflow.org/tensorboard

[opencv-image]: https://img.shields.io/badge/OpenCV-fffcf0?logo=OpenCV&logoColor=100f0f
[opencv-url]: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

[pillow-image]: https://img.shields.io/badge/PIL-fffcf0?logo=python&logoColor=100f0f
[pillow-url]: https://pillow.readthedocs.io/en/stable/

[sklearn-image]: https://img.shields.io/badge/Scikit%20Learn-fffcf0?logo=scikit-learn&logoColor=100f0f
[sklearn-url]: https://scikit-learn.org/stable/

