import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

def main():
    # --- 1. Device Configuration ---
    # Set the device to a GPU if available, otherwise use the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Hyperparameters ---
    # These are the settings we will be tweaking in our experiments
    LEARNING_RATE = 0.01
    BATCH_SIZE = 512
    EPOCHS = 10
    
    # --- 3. Data Loading and Transformation ---
    # For the baseline, we only do the bare minimum: convert images to tensors
    # and normalize them.
    transform = transforms.Compose([
        transforms.ToTensor(),
        # These are the standard normalization values for CIFAR-10
        # Mean and std for CIFAR-10 RGB channels: (R, G, B)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Use a DataLoader for the initial one-time load
    # Note: num_workers=0 can sometimes be fastest for a single pass
    initial_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=0)
    initial_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)

    # --- NEW: Pre-load all data to the GPU ---
    print("Pre-loading data to GPU...")
    start_load_time = time.time()

    # Load all training data
    train_images, train_labels = next(iter(initial_train_loader))
    train_images, train_labels = train_images.to(device), train_labels.to(device)

    # Load all test data
    test_images, test_labels = next(iter(initial_test_loader))
    test_images, test_labels = test_images.to(device), test_labels.to(device)

    print(f"Data pre-loaded in {time.time() - start_load_time:.2f} seconds.")
    
    # --- 4. Model Definition ---
    # A very simple Convolutional Neural Network
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            # Input: 3x32x32
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces size from 32x32 to 16x16
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            # Pool again, from 16x16 to 8x8
            
            # The flattened size will be 64 (channels) * 8 * 8 (image dimension)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10) # 10 output classes

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 64 * 8 * 8) # Flatten the tensor
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleCNN().to(device)

    # --- 5. Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    # Baseline optimizer: SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # --- 6. Training and Validation Loop ---
    start_time = time.time()

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train() # Set the model to training mode
        
        # We shuffle the data manually each epoch
        permutation = torch.randperm(train_images.size(0))
        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, train_images.size(0), BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_images, batch_labels = train_images[indices], train_labels[indices]

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

        # --- Validation (now much simpler) ---
        model.eval()
        correct = 0
        with torch.no_grad():
            # We can evaluate the whole test set in one go since it's small
            outputs = model(test_images)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == test_labels).sum().item()
        
        val_accuracy = 100 * correct / len(test_dataset)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Finished Training. Training loop time: {total_time:.2f} seconds')
    print(f'Final Validation Accuracy: {val_accuracy:.2f}%')

if __name__ == '__main__':
    main()
