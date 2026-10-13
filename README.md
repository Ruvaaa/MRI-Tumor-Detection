# MRI Brain Tumor Detection

A binary image classifier that predicts whether a brain MRI scan shows a **tumor** or **no tumor**, built with a convolutional neural network (CNN) in TensorFlow/Keras.

The full workflow lives in a single notebook, `model.ipynb`: data loading, exploratory analysis, preprocessing and augmentation, model training, and saving the trained model.

> **Disclaimer:** This is a learning/portfolio project. It is not a medical device and must not be used for diagnosis or clinical decisions.

## Overview

| | |
|---|---|
| **Task** | Binary classification (0 = no tumor, 1 = tumor) |
| **Dataset** | Brain Tumor MRI Dataset, 10,093 images |
| **Input** | RGB images resized to 224 × 224 |
| **Framework** | TensorFlow / Keras |
| **Output** | `MRI_Tumor_detection.keras` |

## Dataset

The notebook expects a folder with one subfolder per class:

```
data/
├── no tumor/
└── tumor/
```

Every image is converted to RGB, resized to 224 × 224, and stored in a NumPy array. Folders named `no tumor` get label `0`; everything else gets label `1`. This gives `X` with shape `(10093, 224, 224, 3)` and `Y` with shape `(10093,)`.

## Pipeline

1. **Load data**: read images with Pillow, resize, and build the `X` / `Y` arrays.
2. **Exploratory analysis**: class distribution bar chart, pixel-value histogram, and sample MRI images from each class.
3. **Split**: 75% train, 12.5% validation, 12.5% test (`random_state=42`).
4. **Augmentation** (training set only, via `ImageDataGenerator`): rescale to [0, 1], rotation up to 50°, shear 0.2, zoom 0.2, horizontal flip. The validation set is only rescaled.
5. **Train**: Adam (lr = 0.001), binary cross-entropy, up to 20 epochs, batch size 32.
6. **Evaluate**: accuracy and loss curves for training vs. validation.
7. **Save**: the trained model is written to `MRI_Tumor_detection.keras`.

## Model architecture

```
Input (224, 224, 3)
Conv2D(16, 3x3, ReLU) -> MaxPooling2D
Conv2D(32, 3x3, ReLU) -> MaxPooling2D
Conv2D(64, 3x3, ReLU) -> MaxPooling2D
Dropout(0.2)
Flatten
Dense(12568, ReLU)
Dense(1, sigmoid)
```

Training uses a custom early-stopping callback that halts training once accuracy passes 95%.

## Results

Training stopped after epoch 2 when the 95% accuracy threshold was reached:

| Epoch | Train accuracy | Val accuracy | Val loss |
|---|---|---|---|
| 1 | 87.9% | 94.9% | 0.153 |
| 2 | 95.0% | 92.1% | 0.186 |

The held-out test split is created but not yet evaluated in the notebook, so there is no final test-set metric.

## Requirements

- Python 3.12 (the notebook was run on 3.12.6)
- `tensorflow`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `pillow`
- `jupyter`

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn pillow jupyter
```

## Usage

1. Download the Brain Tumor MRI Dataset and arrange it as shown above.
2. In the data-loading cell of `model.ipynb`, change the hardcoded Windows path (`C:\indianGuyPROJECT\...`) to the location of your `data` folder.
3. Define the `myCallback` class (see Known issues), then run all cells.
4. To use the trained model:

```python
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("MRI_Tumor_detection.keras")

img = Image.open("scan.jpg").convert("RGB").resize((224, 224))
x = np.array(img)[None] / 255.0

prob = model.predict(x)[0][0]
print("Tumor" if prob > 0.5 else "No tumor", f"({prob:.2%})")
```

## Known issues and suggested improvements

- **`myCallback` is not defined** in the notebook, so the training cell fails on a fresh run. A minimal version matching the observed output:
  ```python
  class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
          if logs.get("accuracy", 0) > 0.95:
              print("\nAccuracy is above 95% so cancelling training!")
              self.model.stop_training = True
  ```
- **Hardcoded local path**: use a relative path or a config variable.
- **Very large model**: `Dense(12568)` after `Flatten` produces about 543 million parameters (roughly 2 GB). A smaller dense layer, `GlobalAveragePooling2D`, or transfer learning (e.g. MobileNetV2, EfficientNet) would be far lighter and likely more accurate.
- **Test set never used**: add `model.evaluate` on the test split, plus a confusion matrix, precision, recall, and F1. For tumor detection, recall matters a lot.
- **Unused normalization**: the `X_train / 255.0` cell runs after the generators are built, so it has no effect on training. The generators already rescale.
- **Stopping criterion**: stopping on training accuracy (not validation) is a weak signal. `EarlyStopping(monitor="val_loss", restore_best_weights=True)` is more standard.
- **Stale comments**: a few comments are leftovers (e.g. "Banana_Rotten / Banana_Healthy" in the labeling cell, "256 units" on the dense layer).

## Project structure

```
.
├── model.ipynb                 # full pipeline
├── MRI_Tumor_detection.keras   # saved model (created after training)
└── README.md
```
