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

## Run 6: Ablation Study - Remove RGB Normalization
- **Hypothesis**: The RGB normalization helps improve the accuracy faster.
- **Description**: Comment out the normalization step in the transforms:
   ```
   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
   ```
```
$ python main.py
Using device: cpu
Pre-loading data to GPU...
Data pre-loaded in 7.90 seconds.
Epoch [1/10], Loss: 2.2267, Val Accuracy: 28.43%
Epoch [2/10], Loss: 1.9333, Val Accuracy: 35.69%
Epoch [3/10], Loss: 1.7314, Val Accuracy: 38.89%
Epoch [4/10], Loss: 1.5625, Val Accuracy: 47.16%
Epoch [5/10], Loss: 1.4499, Val Accuracy: 48.79%
Epoch [6/10], Loss: 1.3706, Val Accuracy: 51.85%
Epoch [7/10], Loss: 1.3073, Val Accuracy: 54.01%
Epoch [8/10], Loss: 1.2540, Val Accuracy: 55.60%
Epoch [9/10], Loss: 1.2018, Val Accuracy: 57.66%
Epoch [10/10], Loss: 1.1577, Val Accuracy: 59.99%
Finished Training. Training loop time: 189.21 seconds
Final Validation Accuracy: 59.99%
```
- **Result**: The final accuracy is significantly lower without normalization (59.99% vs. 70.61%), 
  indicating that it plays an important role in training effectiveness.

## Run 7: Experiment 2.1 - Adam Optimizer
- **Hypothesis**: Adam optimizer will converge faster and potentially achieve better final accuracy than SGD.
- **Description**: Switch from SGD to Adam optimizer, adjusting learning rate from 0.01 (SGD) to 0.001 (Adam). Also tested lr=0.0004 based on common Adam defaults.
- **Hardware:** AMD Ryzen 7 5700U with Radeon Graphics, WSL environment.

### Run 7a: Adam, lr=0.001
- **Configuration**: Adam optimizer, lr=0.001, BATCH_SIZE=512, EPOCHS=10
- **Final Validation Accuracy**: 70.54%

```
$ python main.py
Using device: cpu
Pre-loading data...
Data pre-loaded in 14.50 seconds.
Epoch [1/10], Loss: 1.5726, Val Accuracy: 53.48%, Duration: 20.42s
Epoch [2/10], Loss: 1.1831, Val Accuracy: 60.54%, Duration: 20.65s
Epoch [3/10], Loss: 1.0338, Val Accuracy: 63.96%, Duration: 19.86s
Epoch [4/10], Loss: 0.9314, Val Accuracy: 67.05%, Duration: 19.56s
Epoch [5/10], Loss: 0.8551, Val Accuracy: 67.92%, Duration: 19.72s
Epoch [6/10], Loss: 0.7870, Val Accuracy: 69.23%, Duration: 19.96s
Epoch [7/10], Loss: 0.7355, Val Accuracy: 68.72%, Duration: 19.45s
Epoch [8/10], Loss: 0.6935, Val Accuracy: 69.86%, Duration: 19.52s
Epoch [9/10], Loss: 0.6383, Val Accuracy: 71.29%, Duration: 20.05s
Epoch [10/10], Loss: 0.5956, Val Accuracy: 70.54%, Duration: 19.96s
Finished Training. Training loop time: 199.15 seconds
Final Validation Accuracy: 70.54%
```

### Run 7b: Adam, lr=0.0004
- **Configuration**: Adam optimizer, lr=0.0004, BATCH_SIZE=512, EPOCHS=10
- **Final Validation Accuracy**: 66.69%
- **Training Loop Time**: 198.33 seconds
- **Data Pre-Load Before Training**: 13.68 seconds
```
$ python main.py
Using device: cpu
Pre-loading data...
Data pre-loaded in 13.68 seconds.
Epoch [1/10], Loss: 1.6765, Val Accuracy: 49.63%, Duration: 19.54s
Epoch [2/10], Loss: 1.3395, Val Accuracy: 55.69%, Duration: 19.97s
Epoch [3/10], Loss: 1.2108, Val Accuracy: 58.29%, Duration: 20.65s
Epoch [4/10], Loss: 1.1236, Val Accuracy: 60.05%, Duration: 19.99s
Epoch [5/10], Loss: 1.0533, Val Accuracy: 62.89%, Duration: 20.10s
Epoch [6/10], Loss: 0.9957, Val Accuracy: 64.57%, Duration: 19.90s
Epoch [7/10], Loss: 0.9511, Val Accuracy: 65.67%, Duration: 20.55s
Epoch [8/10], Loss: 0.9079, Val Accuracy: 67.06%, Duration: 19.10s
Epoch [9/10], Loss: 0.8725, Val Accuracy: 65.65%, Duration: 18.63s
Epoch [10/10], Loss: 0.8468, Val Accuracy: 66.69%, Duration: 19.90s
Finished Training. Training loop time: 198.33 seconds
Final Validation Accuracy: 66.69%
```
**Analysis**: Adam shows faster initial convergence than SGD, but the final accuracy with lr=0.001 is only comparable to SGD, and lr=0.0004 underperforms. For this simple CNN on CIFAR-10, Adam does not outperform well-tuned SGD in final accuracy; tuning learning rate matters significantly.

## Run 8: Add BatchNorm2d (CPU)
- **Hypothesis**: BatchNorm should stabilize training and improve final accuracy.
- **Description**: Insert BatchNorm2d after each conv layer: Conv -> BN -> ReLU -> Pool. Keep optimizer as plain SGD (no momentum), lr=0.01; BATCH_SIZE=512; EPOCHS=10. Same preloading and normalization as prior runs.
- **Hardware:** AMD Ryzen 7 5700U with Radeon Graphics, WSL environment (CPU).
- **Configuration**: SGD(lr=0.01, momentum=0.0), BATCH_SIZE=512, EPOCHS=10, BatchNorm2d after conv1/conv2.
- **Observation**: Per-epoch time on CPU increased substantially with BN compared to prior CPU runs.

```
$ python main.py
Using device: cpu
Pre-loading data...
Data pre-loaded in 15.01 seconds.
Epoch [1/10], Loss: 1.8820, Val Accuracy: 44.49%, Duration: 29.67s
Epoch [2/10], Loss: 1.5277, Val Accuracy: 49.90%, Duration: 31.31s
Epoch [3/10], Loss: 1.3751, Val Accuracy: 53.62%, Duration: 31.14s
Epoch [4/10], Loss: 1.2801, Val Accuracy: 54.82%, Duration: 33.10s
Epoch [5/10], Loss: 1.2092, Val Accuracy: 56.85%, Duration: 31.33s
Epoch [6/10], Loss: 1.1537, Val Accuracy: 59.29%, Duration: 30.15s
Epoch [7/10], Loss: 1.1083, Val Accuracy: 60.32%, Duration: 32.39s
Epoch [8/10], Loss: 1.0662, Val Accuracy: 61.13%, Duration: 31.49s
Epoch [9/10], Loss: 1.0322, Val Accuracy: 60.05%, Duration: 31.42s
Epoch [10/10], Loss: 1.0021, Val Accuracy: 63.36%, Duration: 32.57s
Finished Training. Training loop time: 314.56 seconds
Final Validation Accuracy: 63.36%
```

**Analysis**: On CPU, adding BatchNorm2d increased per-epoch time substantially and reduced final accuracy relative to the previous CPU baseline (~70%). Likely causes: BN kernels are slower on CPU; momentum was not used (BN often pairs well with SGD+momentum); hyperparameters not re-tuned for BN. Next steps: test SGD+momentum=0.9 with BN, and/or run on GPU where BN is much faster.

### Run 8b: BatchNorm2d + SGD Momentum=0.9 (CPU)
- **Hypothesis**: Adding momentum with BN should improve convergence and recover/beat baseline accuracy.
- **Description**: Same as Run 8 (BN after conv1/conv2), but use SGD with momentum=0.9; lr=0.01; BATCH_SIZE=512; EPOCHS=10.
- **Hardware:** AMD Ryzen 7 5700U with Radeon Graphics, WSL environment (CPU).
- **Configuration**: SGD(lr=0.01, momentum=0.9), BATCH_SIZE=512, EPOCHS=10, BatchNorm2d after conv1/conv2.

```
$ python main.py
Using device: cpu
Pre-loading data...
Data pre-loaded in 13.54 seconds.
Epoch [1/10], Loss: 1.5025, Val Accuracy: 57.68%, Duration: 25.75s
Epoch [2/10], Loss: 1.0837, Val Accuracy: 61.30%, Duration: 30.47s
Epoch [3/10], Loss: 0.9392, Val Accuracy: 66.26%, Duration: 31.51s
Epoch [4/10], Loss: 0.8598, Val Accuracy: 67.09%, Duration: 29.90s
Epoch [5/10], Loss: 0.7828, Val Accuracy: 68.90%, Duration: 29.20s
Epoch [6/10], Loss: 0.7239, Val Accuracy: 67.94%, Duration: 29.94s
Epoch [7/10], Loss: 0.6749, Val Accuracy: 70.57%, Duration: 29.54s
Epoch [8/10], Loss: 0.6353, Val Accuracy: 69.78%, Duration: 30.72s
Epoch [9/10], Loss: 0.5773, Val Accuracy: 69.93%, Duration: 29.87s
Epoch [10/10], Loss: 0.5345, Val Accuracy: 71.83%, Duration: 31.22s
Finished Training. Training loop time: 298.13 seconds
Final Validation Accuracy: 71.83%
```

**Analysis**: Momentum with BN restores and surpasses the earlier CPU baseline, reaching 71.83% final accuracy. However, CPU epoch times remain higher with BN (~29–31s vs ~20s without BN). Next isolation: test plain SGD + momentum=0.9 without BN to quantify momentum’s standalone contribution.

### Run 8c: Remove BatchNorm, keep SGD Momentum=0.9 (CPU)
- **Hypothesis**: Isolate the benefit of momentum alone without BN overhead.
- **Description**: USE_BN=False (no BatchNorm), optimizer=SGD(lr=0.01, momentum=0.9), BATCH_SIZE=512, EPOCHS=10.
- **Hardware:** AMD Ryzen 7 5700U with Radeon Graphics, WSL environment (CPU).
- **Configuration**: Plain SimpleCNN (no BN), SGD(lr=0.01, momentum=0.9), BATCH_SIZE=512, EPOCHS=10.

```
$ python main.py
Using device: cpu
Pre-loading data...
Data pre-loaded in 13.37 seconds.
Epoch [1/10], Loss: 1.8987, Val Accuracy: 45.17%, Duration: 18.02s
Epoch [2/10], Loss: 1.4286, Val Accuracy: 52.15%, Duration: 19.91s
Epoch [3/10], Loss: 1.2547, Val Accuracy: 57.24%, Duration: 21.74s
Epoch [4/10], Loss: 1.1336, Val Accuracy: 60.48%, Duration: 20.84s
Epoch [5/10], Loss: 1.0397, Val Accuracy: 62.27%, Duration: 22.01s
Epoch [6/10], Loss: 0.9639, Val Accuracy: 64.78%, Duration: 22.32s
Epoch [7/10], Loss: 0.8949, Val Accuracy: 66.52%, Duration: 21.32s
Epoch [8/10], Loss: 0.8293, Val Accuracy: 68.32%, Duration: 21.58s
Epoch [9/10], Loss: 0.7829, Val Accuracy: 68.86%, Duration: 21.59s
Epoch [10/10], Loss: 0.7235, Val Accuracy: 70.57%, Duration: 20.55s
Finished Training. Training loop time: 209.88 seconds
Final Validation Accuracy: 70.57%
```

**Analysis**: Momentum alone (no BN) restores accuracy to ~70.6% and keeps epoch time close to the faster, pre-BN runs (~20–22s). Conclusion: on CPU, BN adds notable overhead and didn’t help accuracy in this small CNN, while momentum=0.9 provides a clear benefit with minimal time cost. On GPU, BN would likely be faster and may still help; locally on CPU, prefer plain CNN + momentum for speed/accuracy trade-off.

### Run 8d: No BN, SGD momentum=0.9, weight_decay=5e-4 (CPU)
- **Hypothesis**: Adding standard weight decay should improve generalization slightly over momentum-only.
- **Description**: USE_BN=False, optimizer=SGD(lr=0.01, momentum=0.9, weight_decay=5e-4), BATCH_SIZE=512, EPOCHS=10.
- **Hardware:** AMD Ryzen 7 5700U with Radeon Graphics, WSL environment (CPU).
- **Configuration**: Plain SimpleCNN (no BN), SGD(lr=0.01, momentum=0.9, wd=5e-4), BATCH_SIZE=512, EPOCHS=10.

```
$ python main.py
Using device: cpu
Pre-loading data...
Data pre-loaded in 14.55 seconds.
Epoch [1/10], Loss: 1.8833, Val Accuracy: 43.24%, Duration: 18.46s
Epoch [2/10], Loss: 1.4400, Val Accuracy: 51.42%, Duration: 19.06s
Epoch [3/10], Loss: 1.2625, Val Accuracy: 56.84%, Duration: 19.33s
Epoch [4/10], Loss: 1.1499, Val Accuracy: 59.50%, Duration: 19.92s
Epoch [5/10], Loss: 1.0612, Val Accuracy: 63.00%, Duration: 19.03s
Epoch [6/10], Loss: 0.9716, Val Accuracy: 65.18%, Duration: 19.60s
Epoch [7/10], Loss: 0.9192, Val Accuracy: 65.83%, Duration: 19.68s
Epoch [8/10], Loss: 0.8637, Val Accuracy: 67.28%, Duration: 19.61s
Epoch [9/10], Loss: 0.8058, Val Accuracy: 67.04%, Duration: 19.71s
Epoch [10/10], Loss: 0.7546, Val Accuracy: 69.29%, Duration: 18.89s
Finished Training. Training loop time: 193.29 seconds
Final Validation Accuracy: 69.29%
```

**Analysis**: With weight decay, final accuracy here slightly underperformed momentum-only (~69.3% vs ~70.6%) on this run, though differences can be within run-to-run variance. Time remained in the faster ~19–20s/epoch band without BN. Next, consider trying cosine LR scheduling or tuning weight_decay (e.g., 1e-4 to 5e-4) to see if accuracy improves, or run multiple seeds to reduce variance effects.

### Run 8e: No BN, SGD momentum=0.9 + Cosine LR Schedule (CPU)
- **Hypothesis**: Cosine annealing can improve generalization and final accuracy over a fixed LR.
- **Description**: USE_BN=False, optimizer=SGD(lr=0.01, momentum=0.9), CosineAnnealingLR over 10 epochs, no weight decay; BATCH_SIZE=512, EPOCHS=10.
- **Hardware:** AMD Ryzen 7 5700U with Radeon Graphics, WSL environment (CPU).
- **Configuration**: Plain SimpleCNN (no BN), SGD(lr=0.01, momentum=0.9), cosine schedule (T_max=10), BATCH_SIZE=512, EPOCHS=10.

```
$ python main.py
Using device: cpu
Pre-loading data...
Data pre-loaded in 12.73 seconds.
Epoch [1/10], Loss: 1.8896, Val Accuracy: 45.53%, Duration: 17.93s
Epoch [2/10], Loss: 1.4258, Val Accuracy: 52.42%, Duration: 19.47s
Epoch [3/10], Loss: 1.2589, Val Accuracy: 56.44%, Duration: 19.08s
Epoch [4/10], Loss: 1.1413, Val Accuracy: 59.26%, Duration: 19.20s
Epoch [5/10], Loss: 1.0700, Val Accuracy: 62.27%, Duration: 19.67s
Epoch [6/10], Loss: 0.9929, Val Accuracy: 64.11%, Duration: 20.19s
Epoch [7/10], Loss: 0.9372, Val Accuracy: 65.37%, Duration: 18.95s
Epoch [8/10], Loss: 0.9026, Val Accuracy: 65.57%, Duration: 19.87s
Epoch [9/10], Loss: 0.8781, Val Accuracy: 65.94%, Duration: 20.12s
Epoch [10/10], Loss: 0.8637, Val Accuracy: 66.48%, Duration: 19.34s
Finished Training. Training loop time: 193.83 seconds
Final Validation Accuracy: 66.48%
```

**Analysis**: Cosine schedule underperformed the fixed-LR momentum baseline on this architecture/dataset split (66.5% vs ~70–71%). Likely the LR decay was too aggressive over only 10 epochs with no warmup. If retrying cosine, consider either a higher initial LR (e.g., 0.02), a warmup epoch, or extending total epochs. For now, fixed LR=0.01 with momentum=0.9 (no BN) remains the best CPU speed/accuracy trade-off observed.
