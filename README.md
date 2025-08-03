# Project: CIFAR-10 Speedrun 🚀

This repository is dedicated to learning and building intuition for neural networks by "speedrunning" the CIFAR-10 dataset.

## TL;DR (Quick Start)
- Single script: main.py
- Default strong CPU baseline: SimpleCNN, SGD(lr=0.02, momentum=0.9), 10 epochs, no BN/augmentation
- Reproducible toggles in code:
  - USE_BN = False  # add BatchNorm if True
  - USE_COSINE = False  # cosine LR schedule over EPOCHS if True
- Latest best CPU result (no aug): 71.87% in ~197s (see experiments.md Run 8f)
- All experiment logs with per-epoch metrics: experiments.md

Run locally:
```
python main.py
```

## The Philosophy: Learn by Iterating, Fast

We shorten the idea-to-result loop to minutes. CIFAR-10 plus a tiny CNN gives fast feedback so you can isolate one change at a time and build intuition.

The core idea, inspired by practitioners like Keller Jordan, is that the speed of learning is limited by the time it takes to go from a new idea to a concrete result. Traditional machine learning workflows can involve long training times, making it difficult to connect a specific change to its outcome.

This project flips that on its head. By using a small dataset like CIFAR-10 and a simple, fast-to-train model, we can reduce training time from hours to **seconds**. This creates a tight feedback loop that allows for massive experimentation. The goal isn't to achieve state-of-the-art accuracy on the first try, but to run hundreds of small, targeted experiments to see what works, what doesn't, and most importantly, *why*.

Through this process of rapid, iterative experimentation, we aim to build a deep, practical intuition for the fundamental concepts of machine learning.

## The Dataset: CIFAR-10

CIFAR-10 is a classic computer vision dataset consisting of 60,000 32x32 color images in 10 distinct classes:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

## Current Onboarding: What this repo gives you
1) A runnable baseline that trains end-to-end in a few minutes on CPU.
2) A minimal model with switches to compare common techniques (BatchNorm, cosine schedule).
3) A living lab notebook: experiments.md with hypotheses, configs, timings, and per-epoch accuracy.

## How to reproduce the main baselines
- Baseline (fast, CPU-friendly):
  - In main.py: USE_BN=False, USE_COSINE=False, LEARNING_RATE=0.02, momentum=0.9
  - Command: `python main.py`
  - Expected: ~71–72% in ~19–20s/epoch on CPU
- BatchNorm variant (slower on CPU, faster on GPU):
  - USE_BN=True, momentum=0.9, lr=0.01 (or retune)
  - On CPU, BN increases epoch time. On GPU, BN is much faster and can help accuracy.
- Cosine schedule:
  - USE_COSINE=True with fixed EPOCHS; consider warmup or longer training for benefit. Recent run with lr=0.02, momentum=0.9, cosine over 10 epochs achieved 71.78% on CPU.

## Experiment Summary

A detailed log of all experiments is available in `experiments.md`. Here is a high-level summary of the key findings:

*   **Initial Baseline (Runs 1-6):** Established a simple CNN baseline, first on CPU (~69%) and then GPU (~71%). Early experiments showed massive speedups from optimizing the data loading pipeline (pre-loading and normalization), reducing training loop time from ~287s to ~14s on GPU.
*   **SimpleCNN Tuning (Runs 7-8):** Found that for a small CNN, well-tuned SGD with momentum (lr=0.02, momentum=0.9) outperformed Adam and was the best configuration on CPU, achieving **71.87%** accuracy. BatchNorm slowed down CPU execution without providing a clear accuracy benefit.
*   **Architecture Change to ResNet-9 (Run 9):** Switching to a ResNet-9 model on a GPU immediately boosted accuracy to **75.54%**, demonstrating the power of a better architecture.
*   **The Power of Small Batches (Runs 10-17):** A systematic sweep of batch sizes from 512 down to 16 revealed a surprising trend: smaller batch sizes consistently improved accuracy. Batch size 32 achieved **87.18%**, while batch size 64 hit **85.69%** in just 10 epochs.
*   **Reaching 90% Accuracy (Run 18):** By extending training to 20 epochs with a cosine learning rate schedule and the optimal batch size of 64, the model reached **90.68%** accuracy.
*   **Advanced Optimizations (Runs 19-26):**
    *   **Learning Rate:** Found an optimal LR of 0.04 for the ResNet-9 setup, hitting **88.27%** in 10 epochs.
    *   **Mixed Precision (AMP):** Using AMP provided a **~29% speedup** on a T4 GPU with no loss in accuracy.
    *   **Hardware:** An A100 GPU was ~2.7x faster than a T4.
    *   **Augmentations:** Adding `ColorJitter` provided a slight accuracy boost, while `TrivialAugmentWide` and `RandomRotation` were less effective than the baseline augmentations for this setup.

The final recommended configuration for a fast, high-accuracy 10-epoch run is ResNet-9, batch size 64, LR=0.04, cosine schedule, and AMP enabled.

## Suggested next experiments
- Add simple augmentation (RandomCrop(32, padding=4), RandomHorizontalFlip) to push >75% without changing model depth.
- GPU pass: re-evaluate BN on GPU (much faster kernels). Expect better speed and potentially accuracy.
- LR/momentum small sweep around the current best: lr ∈ {0.015, 0.02, 0.025}, momentum ∈ {0.8, 0.9, 0.95}.
- Optional optimizer baseline: AdamW(lr=1e-3, wd=5e-4) for completeness.

### Pushing the Limits
Once you are comfortable with the above, try more advanced experiments:
- Architectural Changes: Add dropout, increase the model depth or width, or try a simple ResNet block.
- Advanced Optimizers: Try newer optimizers like LAdam or RAdam.
- CutMix / MixUp: Implement more advanced data augmentation strategies.

## Environment setup
- Python 3.10+
- pip install torch torchvision (or use Conda)
- GPU optional but recommended for BN and future augmentation-heavy experiments

## Why this structure?
- One-file training loop lowers friction and highlights cause/effect of each change.
- experiments.md documents “what happened and why” so you can learn from every run.
- Small steps, real measurements, fast iteration.

**Happy Speedrunning!**
