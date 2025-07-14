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

    dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    dataset_test  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    loader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
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
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(loader_train):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # --- Validation ---
        model.eval() # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad(): # We don't need to calculate gradients during validation
            for inputs, labels in loader_test:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(loader_train):.4f}, Val Accuracy: {val_accuracy:.2f}%')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Finished Training. Total time: {total_time:.2f} seconds')
    print(f'Final Validation Accuracy: {val_accuracy:.2f}%')

if __name__ == '__main__':
    main()
