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
    LEARNING_RATE = 0.02  # Higher LR test for SGD momentum baseline
    BATCH_SIZE = 512
    EPOCHS = 10
    USE_BN = False  # Toggle BatchNorm on/off to isolate its effect
    USE_COSINE = True  # Enable cosine annealing LR schedule
    
    print("Pre-loading data...")
    start_load_time = time.time()

    # --- 3. Data Loading and Transformation ---
    # For the baseline, we only do the bare minimum: convert images to tensors
    # and normalize them.
    transform = transforms.Compose([
        transforms.ToTensor(),
        # These are the standard normalization values for CIFAR-10
        # Mean and std for CIFAR-10 training set RGB channels: (R, G, B)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Use a DataLoader for the initial one-time load
    # Note: num_workers=0 can sometimes be fastest for a single pass
    initial_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=0)
    initial_test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=len(test_dataset),  shuffle=False, num_workers=0)

    # Load all data
    train_images, train_labels = next(iter(initial_train_loader))
    test_images,  test_labels  = next(iter(initial_test_loader))
 
    # Send to device
    train_images, train_labels = train_images.to(device), train_labels.to(device)
    test_images,   test_labels = test_images.to(device),  test_labels.to(device)

    print(f"Data pre-loaded in {time.time() - start_load_time:.2f} seconds.")
    
    # This is a real Python/PyTorch mystery, when testing locally in WSL with CPU (no GPU):
    #  - If we load these 4 variables and use them, the Epoch time is fast (18s).
    #  - If we load and then save these 4 variables, delete the vars, and reload them from the files, the Epoch time is fast (18s)!
    #  - But if we skip the loader and instead just load the 4 variables from the files, the epoch time is SLOW (29s, 27s).
    #  - If we load but skip save, and then reload the 4 variables from older files, epoch time is FAST (19, 23, 23s).
    # What in the world is going on???? Why is the epoch time slow unless the loader code runs?
    # It seems unnecessary if the 4 vars are loaded from files.
    # The good news is: this issue doesn't exist when using a GPU, specifically the T4 in Google Colab.
    # With the T4, the epoch times were like 1.3 seconds even if I skipped the loader and simply read the 4 variables from files.
    
    # # --- NEW: Save the pre-loaded tensors to files ---
    # print("Saving pre-loaded tensors to disk...")
    # torch.save(train_images, './data/train_images.pt')
    # torch.save(train_labels, './data/train_labels.pt')
    # torch.save(test_images, './data/test_images.pt')
    # torch.save(test_labels, './data/test_labels.pt')
    # print("Tensors saved.")

    # --- NEW: Load the tensors back from files ---
    # To demonstrate, we'll clear the variables and load them from the saved files
    # del train_images, train_labels, test_images, test_labels
    # print("Loading tensors from disk...")
    # start_load_time = time.time()
    # train_images = torch.load('./data/train_images.pt', map_location=device).float().clone(memory_format=torch.contiguous_format)
    # train_labels = torch.load('./data/train_labels.pt', map_location=device).clone(memory_format=torch.contiguous_format)
    # test_images  = torch.load('./data/test_images.pt',  map_location=device).float().clone(memory_format=torch.contiguous_format)
    # test_labels  = torch.load('./data/test_labels.pt',  map_location=device).clone(memory_format=torch.contiguous_format)
    # print(f"Tensors loaded from disk in {time.time() - start_load_time:.2f} seconds.")
    
    # --- 4. Model Definition ---
    # A very simple Convolutional Neural Network
    class SimpleCNN(nn.Module):
        def __init__(self, use_bn: bool = True):
            super(SimpleCNN, self).__init__()
            self.use_bn = use_bn
            # Input: 3x32x32
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
            if self.use_bn:
                self.bn1 = nn.BatchNorm2d(32)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces size from 32x32 to 16x16
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
            if self.use_bn:
                self.bn2 = nn.BatchNorm2d(64)
            # Pool again, from 16x16 to 8x8
            
            # The flattened size will be 64 (channels) * 8 * 8 (image dimension)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10) # 10 output classes

        def forward(self, x):
            x = self.conv1(x)
            if self.use_bn:
                x = self.bn1(x)
            x = torch.relu(x)
            x = self.pool(x)

            x = self.conv2(x)
            if self.use_bn:
                x = self.bn2(x)
            x = torch.relu(x)
            x = self.pool(x)

            x = x.view(-1, 64 * 8 * 8) # Flatten the tensor
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleCNN(use_bn=USE_BN).to(device)

    # --- 5. Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    # Plain SGD with momentum=0.9 (no weight decay) for isolation (BN toggled by USE_BN)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    # Optional cosine LR schedule across total epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) if USE_COSINE else None

    # --- 6. Training and Validation Loop ---
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
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

        # Step LR scheduler once per epoch (after training)
        if scheduler is not None:
            scheduler.step()

        # --- Validation (now much simpler) ---
        model.eval()
        correct = 0
        with torch.no_grad():
            # We can evaluate the whole test set in one go since it's small
            outputs = model(test_images)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == test_labels).sum().item()
        
        val_accuracy = 100 * correct / test_images.size(0)
        epoch_duration = time.time() - epoch_start_time
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Duration: {epoch_duration:.2f}s')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Finished Training. Training loop time: {total_time:.2f} seconds')
    print(f'Final Validation Accuracy: {val_accuracy:.2f}%')

if __name__ == '__main__':
    main()
