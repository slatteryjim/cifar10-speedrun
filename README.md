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

Refer to experiments.md for exact runs:
- Adam vs SGD (Run 7a/7b): Adam did not beat tuned SGD in final accuracy for this small CNN.
- BN on CPU (Run 8/8b): BN without momentum underperformed; BN+momentum improved accuracy but was slower on CPU.
- Momentum alone (Run 8c): Solid gains with low overhead; best CPU trade-off.
- Weight decay and cosine (Run 8d/8e): Did not beat fixed LR momentum in 10 epochs.
- Raised LR (Run 8f): lr=0.02 with momentum=0.9 achieved 71.87% on CPU without BN.

## Suggested next experiments
- Add simple augmentation (RandomCrop(32, padding=4), RandomHorizontalFlip) to push >75% without changing model depth.
- GPU pass: re-evaluate BN on GPU (much faster kernels). Expect better speed and potentially accuracy.
- LR/momentum small sweep around the current best: lr ∈ {0.015, 0.02, 0.025}, momentum ∈ {0.8, 0.9, 0.95}.
- Optional optimizer baseline: AdamW(lr=1e-3, wd=5e-4) for completeness.

## Environment setup
- Python 3.10+
- pip install torch torchvision (or use Conda)
- GPU optional but recommended for BN and future augmentation-heavy experiments

## Why this structure?
- One-file training loop lowers friction and highlights cause/effect of each change.
- experiments.md documents “what happened and why” so you can learn from every run.
- Small steps, real measurements, fast iteration.

**Happy Speedrunning!**
