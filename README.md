# Project: CIFAR-10 Speedrun 🚀

This repository is dedicated to learning and building intuition for neural networks by "speedrunning" the CIFAR-10 dataset.

## The Philosophy: Learn by Iterating, Fast

The core idea, inspired by practitioners like Keller Jordan, is that the speed of learning is limited by the time it takes to go from a new idea to a concrete result. Traditional machine learning workflows can involve long training times, making it difficult to connect a specific change to its outcome.

This project flips that on its head. By using a small dataset like CIFAR-10 and a simple, fast-to-train model, we can reduce training time from hours to **seconds**. This creates a tight feedback loop that allows for massive experimentation. The goal isn't to achieve state-of-the-art accuracy on the first try, but to run hundreds of small, targeted experiments to see what works, what doesn't, and most importantly, *why*.

Through this process of rapid, iterative experimentation, we aim to build a deep, practical intuition for the fundamental concepts of machine learning.

## The Dataset: CIFAR-10

CIFAR-10 is a classic computer vision dataset consisting of 60,000 32x32 color images in 10 distinct classes:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

It's the perfect "gym" for this project—complex enough to be interesting, but small enough to enable lightning-fast training on a modern GPU.

## The Plan of Attack

We will approach this in a structured, scientific manner. For each change, we will form a hypothesis, run the experiment, and analyze the results.

### Phase 0: Setup

1.  **Environment**: Create a Python environment. This project will use `PyTorch` for its flexibility and `torchvision` for easy data loading. A GPU is highly recommended (Google Colab is a great free option).
2.  **Experiment Tracking**: While you can start with notebooks, it's highly recommended to use a tool like **Weights & Biases** to log and compare results. It's free for personal use and invaluable for this kind of high-iteration workflow.
3.  **Codebase**: Create a clean, simple Python script (`main.py`) that handles data loading, model definition, training, and validation. The goal is to have a single place to make quick modifications.

### Phase 1: The Baseline

The first goal is to establish a simple, fast, and reliable baseline. This is the "control" against which all future experiments will be measured.

*   **Action**:
    *   **Model**: A simple CNN with 2-3 convolutional layers, max pooling, and 1-2 fully connected layers.
    *   **Optimizer**: `torch.optim.SGD` (Stochastic Gradient Descent) with a fixed learning rate (e.g., `0.01`) and momentum (e.g., `0.9`).
    *   **Data Augmentation**: None. Just a simple tensor conversion and normalization.
    *   **Epochs**: 10 epochs.
*   **Hypothesis**:
    *   **Reasoning**: This is the most basic setup. SGD is a workhorse optimizer, but without any learning rate scheduling or data augmentation, the model will likely learn the training set reasonably well but won't generalize perfectly to the test set. It might also get stuck or converge slowly.
    *   **Expected Result**: Training should complete in under 60 seconds. We'll likely see an accuracy of around **70-75%**. The validation loss will probably start to flatten or even increase towards the end, indicating the onset of overfitting.

### Phase 2: Systematic Experimentation

Now, we will change **one variable at a time** and compare it to the baseline.

#### Experiment 2.1: The Optimizer

*   **Action**: Change the optimizer from `SGD` to `torch.optim.Adam`. Keep the learning rate the same initially (`0.01`).
*   **Hypothesis**:
    *   **Reasoning**: Adam is an adaptive optimizer that maintains a per-parameter learning rate. It often converges much faster than SGD in the initial epochs. However, its aggressive nature can sometimes lead it to a sharper, less generalizable minimum.
    *   **Expected Result**: The model will train faster, achieving a higher accuracy in fewer epochs. The final accuracy might be slightly higher or lower than SGD, but the "area under the curve" for the accuracy plot will be much better. We might need to lower the learning rate, as Adam can be sensitive to high initial rates.

#### Experiment 2.2: Batch Normalization

*   **Action**: Revert to the baseline `SGD` optimizer. Add `nn.BatchNorm2d` layers after each convolutional layer in the model.
*   **Hypothesis**:
    *   **Reasoning**: Batch Normalization stabilizes the network by normalizing the inputs to each layer. This has a regularizing effect and allows for much higher learning rates without the training becoming unstable.
    *   **Expected Result**: A significant improvement in both training speed and final accuracy. The model should train much more stably, and the final accuracy should jump by **5-10%**. This will likely be one of the most impactful changes we make.

#### Experiment 2.3: Data Augmentation

*   **Action**: Using the Batch Norm model from the previous step, add `transforms.RandomHorizontalFlip` and `transforms.RandomCrop` (with padding) to the training data transformations.
*   **Hypothesis**:
    *   **Reasoning**: Data augmentation artificially expands the dataset by creating modified versions of the training images. This forces the model to learn more robust and general features, making it less likely to overfit.
    *   **Expected Result**: Training time per epoch will increase slightly due to the transformations. The training accuracy might increase more slowly, but the **validation accuracy should be significantly higher** and more stable over time. This is a powerful regularization technique. The gap between training and validation loss should be smaller.

#### Experiment 2.4: Learning Rate Scheduling

*   **Action**: Using the best model so far (with Batch Norm and Augmentation), add a learning rate scheduler, such as `torch.optim.lr_scheduler.OneCycleLR`. This scheduler starts with a low learning rate, warms up to a high one, and then anneals down.
*   **Hypothesis**:
    *   **Reasoning**: A fixed learning rate is a compromise. It's often too high at the end of training (preventing convergence to the best minimum) and could be higher at the beginning. `OneCycleLR` is known to provide very fast convergence and excellent regularization.
    *   **Expected Result**: This should provide the best results yet. We should see faster convergence and potentially reach a new peak accuracy, possibly **over 90%**, even within just 10-20 epochs.

### Phase 3: Pushing the Limits

Once you are comfortable with the above, try more advanced experiments:

*   **Architectural Changes**: Add dropout, increase the model depth or width, or try a simple ResNet block.
*   **Advanced Optimizers**: Try newer optimizers like `LAdam` or `RAdam`.
*   **CutMix / MixUp**: Implement more advanced data augmentation strategies.

By the end of this process, you will not just have a model that classifies images; you will have a deep, intuitive sense of what makes a neural network learn effectively.

**Happy Speedrunning!**
