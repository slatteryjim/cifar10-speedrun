# Experiment Results: CIFAR-10 Speedrun

## Run 1: Baseline

**Description:** The very first run with a simple CNN, basic SGD optimizer, and no data augmentation.

**Hardware:** Run performed on CPU (no GPU used), AMD Ryzen 7 5700U with Radeon Graphics, from within a WSL environment.

**Total Training Time:** 287.10 seconds
**Final Validation Accuracy:** 69.39%

**Configuration:**
| Hyperparameter | Value      |
| :---           | :---       |
| Model          | SimpleCNN  |
| Optimizer      | SGD        |
| Learning Rate  | 0.01       |
| Batch Size     | 512        |
| Epochs         | 10         |
| Augmentation   | None       |

**Per-Epoch Validation Accuracy:**
| Epoch | Loss   | Accuracy (%) |
| :---  | :---   | :---         |
| 1     | 1.9221 | 43.06        |
| 2     | 1.4822 | 49.47        |
| 3     | 1.2928 | 55.78        |
| 4     | 1.1590 | 59.61        |
| 5     | 1.0672 | 61.38        |
| 6     | 0.9718 | 64.61        |
| 7     | 0.9079 | 66.09        |
| 8     | 0.8463 | 67.30        |
| 9     | 0.7878 | 68.78        |
| 10    | 0.7337 | 69.39        |

---

Add new experiment results below as you run more tests!

## Run 2: Baseline (Google Colab, T4 GPU)

**Description:** Baseline run with a simple CNN, basic SGD optimizer, and no data augmentation.

**Hardware:** Google Colab (Python 3 Google Compute Engine backend), T4 GPU.

**Total Training Time:** 131.24 seconds
**Final Validation Accuracy:** 70.77%

**Configuration:**
| Hyperparameter | Value      |
| :---           | :---       |
| Model          | SimpleCNN  |
| Optimizer      | SGD        |
| Learning Rate  | 0.01       |
| Batch Size     | 512        |
| Epochs         | 10         |
| Augmentation   | None       |

**Per-Epoch Validation Accuracy:**
| Epoch | Loss   | Accuracy (%) |
| :---  | :---   | :---         |
| 1     | 1.8925 | 43.67        |
| 2     | 1.4129 | 53.29        |
| 3     | 1.2523 | 57.70        |
| 4     | 1.1367 | 61.57        |
| 5     | 1.0462 | 64.08        |
| 6     | 0.9601 | 66.00        |
| 7     | 0.8875 | 66.69        |
| 8     | 0.8394 | 68.57        |
| 9     | 0.7809 | 70.04        |
| 10    | 0.7217 | 70.77        |


## Run 3: Increase DataLoaders
- **Hypothesis**: Increasing `num_workers` from 2 to 4 will reduce the data loading bottleneck, allowing the GPU to be fed faster and thus decreasing the total training time. Accuracy should be nearly identical.
- **Description**: Same as baseline, but with `num_workers=4`.
- **Total Training Time**: 135.00 seconds
- **Final Validation Accuracy**: 69.56%
- **Configuration**: `num_workers=4`, `BATCH_SIZE=512`, `SGD(lr=0.01)`
- **Result**: Didn't help on Google Colab T4 GPU. Logged warning:
    ```
    warnings.warn(
    /usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py:624: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
    ```

## Run 4: Pre-populate and Normalize the Data before creating DataLoaders
- **Hypothesis**: The DataLoaders are a bottleneck. Repeatedly loading and normalizing the data.
- **Description**: Pre-populate the dataset and normalize it in tensor form before creating DataLoaders. 
  However, we're still loading each batch onto the device during training.
- **Hardware:** Google Colab (Python 3 Google Compute Engine backend), T4 GPU.
- **Configuration**: `num_workers=2`, `BATCH_SIZE=512`, `SGD(lr=0.01)`
- **Result**: Huge speedup! The repeated loading and normalizing of the data into tensor form was a huge bottleneck!
- **Total Training Loop Time**: 23.31 seconds (!!!)
- **Final Validation Accuracy**: 69.64%

## Run 5: Pre-load and Normalize the Data and Move it all to the Device
- **Hypothesis**: Repatedly sending the batches to the device is a bottleneck.
- **Description**: Crate the DataLoaders in one huge batch and load it onto the device before training.
- **Hardware:** Google Colab (Python 3 Google Compute Engine backend), T4 GPU.
- **Configuration**: `num_workers=2`, `BATCH_SIZE=512`, `SGD(lr=0.01)`
- **Result**: Further speedup! The repeated sending of the batches to the device was a bottleneck!
- **Data Pre-Load Before Training**: 15.24 seconds
- **Total Training Loop Time**: 13.73 seconds (!!!)
- **Final Validation Accuracy**: 69.81%

Same code run locally on desktop (CPU):
- **Hardware:** AMD Ryzen 7 5700U with Radeon Graphics, WSL environment.
- **Data Pre-Load Before Training**: 12.87 seconds
- **Total Training Loop Time**: 195.08 seconds
- **Final Validation Accuracy**: 70.61%