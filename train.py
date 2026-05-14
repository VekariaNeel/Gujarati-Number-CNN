import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 30        
img_size = 256

# ─────────────────────────────────────────────
# DATA AUGMENTATION (train only)
# Handwritten digits vary in rotation, size, position.
# Augmentation teaches the model to handle that variation.
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(15),               # Slight rotation variance
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),  # Shift & scale
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Lighting variance
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# No augmentation for test — only normalize
test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dir = "Train Set"
test_dir = "Test Set"

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset  = datasets.ImageFolder(root=test_dir,  transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# BatchNorm     → stabilizes & speeds up training
# Dropout       → prevents overfitting
# 5th Conv      → more feature extraction capacity
# Wider FC head → more representational power
# ─────────────────────────────────────────────
class GujaratiDigitCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(GujaratiDigitCNN, self).__init__()

        # Each block: Conv → BN → ReLU → Pool
        # BatchNorm normalizes activations so gradients flow better
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)          # 256 → 128
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)          # 128 → 64
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)          # 64 → 32
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)          # 32 → 16
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)          # 16 → 8
        )

        # Output is always 4×4 regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),           # Randomly zeros neurons → prevents co-adaptation
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GujaratiDigitCNN(num_classes=len(train_dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  

# AdamW properly decouples weight decay from gradient update → better generalization
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# ReduceLROnPlateau: if val loss stops improving → reduce LR automatically
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
def train():
    best_test_acc = 0.0
    print(f"Starting training on {device}...")
    print(f"Classes: {train_dataset.classes}\n")

    for epoch in range(epochs):
        # ── Train Phase ──
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping → prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc  = 100 * correct / total

        # ── Eval Phase (every epoch) ──
        test_acc = evaluate(verbose=False)

        # Step scheduler based on test accuracy
        scheduler.step(test_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{epochs}] | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        # Saves best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "gujarati_digit_model_best.pth")
            print(f"  ✅ New best model saved ({best_test_acc:.2f}%)")

    print(f"\nTraining complete. Best Test Accuracy: {best_test_acc:.2f}%")


# ─────────────────────────────────────────────
# Evaluation Loop
# ─────────────────────────────────────────────
def evaluate(verbose=True):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    if verbose:
        print(f"Final Test Accuracy: {acc:.2f}%")
    return acc


if __name__ == "__main__":
    train()
    # Load best checkpoint and do final eval
    model.load_state_dict(torch.load("gujarati_digit_model_best.pth"))
    evaluate()