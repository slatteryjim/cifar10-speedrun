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
