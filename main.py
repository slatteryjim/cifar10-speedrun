import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import os

def main():
    # --- 1. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = torch.cuda.is_available()
    print(f"Using device: {device}")
    if USE_AMP:
        print("Using Automatic Mixed Precision (AMP).")

    # --- 2. Hyperparameters ---
    LEARNING_RATE = 0.04  # Higher LR for 10 epochs + cosine
    BATCH_SIZE = 64
    EPOCHS = 10
    USE_BN = True  # ResNet typically benefits from BN
    USE_COSINE = True
    
    print("Pre-loading data...")
    start_load_time = time.time()

    # --- 3. Data Loading and Transformation ---
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_transform)
    test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Create data loaders for batch training
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,  # This helps speed up data transfer to GPU
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,  # Use same batch size for consistency
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    print(f"Data loaders created in {time.time() - start_load_time:.2f} seconds.")
    
    # --- 4. ResNet-9 Model Definition ---
    class BasicBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    class ResNet9(nn.Module):
        def __init__(self, num_classes=10):
            super(ResNet9, self).__init__()
            self.in_channels = 64
            
            # Initial conv layer
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            
            # Residual blocks
            self.layer1 = self._make_layer(64, 2, stride=1)
            self.layer2 = self._make_layer(128, 2, stride=2)
            self.layer3 = self._make_layer(256, 2, stride=2)
            self.layer4 = self._make_layer(512, 2, stride=2)
            
            # Global average pooling and classifier
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

            # Initialize weights
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def _make_layer(self, out_channels, blocks, stride):
            downsample = None
            if stride != 1 or self.in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )

            layers = []
            layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels
            for _ in range(1, blocks):
                layers.append(BasicBlock(out_channels, out_channels))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

    model = ResNet9().to(device, memory_format=torch.channels_last)
    # model = torch.compile(model)  # This 100 second penalty was not worthwhile at all in our 10 epoch test
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # --- 5. Loss Function and Optimizer ---
    print(f"Config: LR={LEARNING_RATE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, COSINE={USE_COSINE}, AMP={USE_AMP}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) if USE_COSINE else None

    # --- 6. Training and Validation Loop ---
    scaler = torch.amp.GradScaler(device, enabled=USE_AMP)
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        # --- Training ---
        model.train()
        epoch_loss_tensor = torch.tensor(0.0, device=device)
        num_batches = 0
        
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device, non_blocking=True, memory_format=torch.channels_last)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Use autocast for mixed precision
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)

            # Scaler handles the backward pass and optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_tensor += loss.detach()
            num_batches += 1

        avg_loss = (epoch_loss_tensor / num_batches).item() if num_batches > 0 else 0.0

        if scheduler is not None:
            scheduler.step()

        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for test_images, test_labels in test_loader:
                test_images = test_images.to(device, non_blocking=True, memory_format=torch.channels_last)
                test_labels = test_labels.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
                    outputs = model(test_images)
                _, predicted = torch.max(outputs.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
        
        val_accuracy = 100 * correct / total
        epoch_duration = time.time() - epoch_start_time
        total_elapsed = time.time() - start_time
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Duration: {epoch_duration:.2f}s, Total: {total_elapsed:.1f}s')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Finished Training. Training loop time: {total_time:.2f} seconds')
    print(f'Final Validation Accuracy: {val_accuracy:.2f}%')

if __name__ == '__main__':
    main()
