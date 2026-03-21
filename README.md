# Offline Signature Verification
![header](assets/header.png)

> Full [documentation!](https://potatogarden.surge.sh/AI--and--Deep-Learning/Offline-Signature-Verification)
>
> [Test the model here!](https://sct-signature-demo.streamlit.app/) (*psst, it's stateless*)

This OSV model utilises [EfficientNetV2](https://arxiv.org/abs/2104.00298) as the feature extraction backbone. The backbone is complemented with a custom loss function, $L_{SC+}$, built on [Hard negative examples are hard, but useful](https://arxiv.org/abs/2007.12749). This project also features an extension of $PK$ sampling, $PKFM$, providing more control over the sampling of intra-class and inter-class signatures during training.

## Model Performance

> The model is test on a held-out test set spliced from CEDAR. 

| Metrics (Threshold: 0.725) | Results | 
| -------------------------- | ------- |
| Accuracy	                 | 84.85%  |
| True Positive Rate	     | 84.80%  |
| False Positive Rate	     | 15.20%  |
| AUC	                     | 0.9284  |
| EER	                     | 15.15%  |
	
| Average Similarity Score (Cosine Similarity) | Results |
| -------------------------------------------- | ------- |
| Positive Score	                           | 0.8444  |
| Negative Score	                           | 0.3644  |

## Key Features

- The loss function is built on [Hard negative examples are hard, but useful](https://arxiv.org/abs/2007.12749), improving the triplet optimisation and computation comsumption. $L_{SC+}$ adapts the loss function to this domain by introducing a positive boundary. 
- $PKFM$ ensures that sufficient hard forgeries are shown in each batch during training. This is useful when the dataset is imbalanced. 


## Built With

| Components                                           |  Version      |  
|------------------------------------------------------|---------------|
| [![Python][python-image]][python-url]                | `3.13.9`      |
| [![PyTorch][pytorch-image]][pytorch-url]             | `2.9.0+cu130` |
| [![NumPy][numpy-image]][numpy-url]                   | `2.2.6`       |
| [![TensorBoard][tensorboard-image]][tensorboard-url] | `2.20.0`      |
| [![OpenCV][opencv-image]][opencv-url]                | `4.12.0.88`   |
| [![PIL][pillow-image]][pillow-url]                   | `11.3.0`      |
| [![Scikit Learn][sklearn-image]][sklearn-url]        | `1.7.2`       |
| [![Matplotlib][matplotlib-image]][matplotlib-url]    | `3.10.7`      |
| [![Seaborn][seaborn-image]][seaborn-url]             | `0.13.2`      |

## Installation

Clone the repository to your selected directory and start training!

```bash
git clone https://github.com/HappyPotatoHead/signature-verification-sct-plus
```

## Usage example

### Configurations

Edit these settings to modify the training pipeline

#### Signature images

```python

# If the dataset is downloadable online
# This particular dataset is hosted on Kaggle
DATASET_SOURCE: Dict[str, str] = {
    "cedar": "shreelakshmigp/cedardataset", 
}

# Directory containing unprocessed signature images
DATASET_PATH: Dict[str, str] = {
    "CEDAR": "data\\CEDAR"
}

IMAGE_FORMATS: List[str] = [".png", ".jpg", ".jpeg", ".bmp"]
```

```python
# Data Augmentations
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

    # Referred directory for tensorboard
    "LOG_DIR": "runs/exp_01_sct"
}

# Optimiser Algorithm
OPTIMISER_PARAMS: Dict[str, str | float] = {
    "optimiser": "AdamW",
    "weight_decay": 1e-3,
}

# Linear learning followed by cosine annealing
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

## Training

Switch between loss function here. 

```python
loss_function = SCTLossWrapper(
    method = "sct", 
    lam = 1.0, 
    margin = 0.5, 
    positive_pull_weight = 0.5, 
    verbose = True
)
# loss_function = TripletLoss(0.5,"batch_hard")
# loss_function = TripletLoss(0.5,"batch_semi_hard")

model = FeatureExtractionModel("efficientnet_v2_m", 256, "IMAGENET1K_V1")
```

## Development setup

1. Navigate to the project directory
2. Create a virtual environment, preferably with a package and environment manager. (I use mini forge)
3. Run
    ```bash
    pip install -r requirements.txt
    ```
4. Navigate to `main.py` to start training and testing the model. 


## Release History

* 1.0
    * First release! 
* 1.01 
    * README minor fixes

## Meta

Jimmy Ding | [Potato Garden](https://potatogarden.surge.sh/) | [Github](https://github.com/HappyPotatoHead/) | jimmydingjk@gmail.com

Distributed under the Apache license. See `LICENSE` for more information.

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