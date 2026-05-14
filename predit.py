import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

# =====================================================
# MODEL ARCHITECTURE
# MUST MATCH TRAINING CODE EXACTLY
# =====================================================

class GujaratiDigitCNN(nn.Module):

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(GujaratiDigitCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),

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


# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# =====================================================
# LOAD MODEL
# =====================================================

model = GujaratiDigitCNN(num_classes=10).to(device)

model.load_state_dict(
    torch.load("gujarati_digit_model_best.pth", map_location=device)
)

model.eval()

print("Model loaded successfully!")


# =====================================================
# IMAGE TRANSFORM
# MUST MATCH TEST TRANSFORM
# =====================================================

transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.Grayscale(num_output_channels=1),

    transforms.ToTensor(),

    transforms.Normalize((0.5,), (0.5,))
])


# =====================================================
# CLASS LABELS
# =====================================================

classes = ['0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9']


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_image(image_path):

    # Open image
    image = Image.open(image_path)

    print(f"\nOriginal Image Size: {image.size}")

    # Convert grayscale
    image = image.convert("L")

    # OPTIONAL:
    # Uncomment if image colors are reversed

    # image = ImageOps.invert(image)

    # Apply transform
    image = transform(image)

    print(f"Tensor Shape: {image.shape}")

    # Add batch dimension
    image = image.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_digit = classes[predicted.item()]
    confidence_score = confidence.item() * 100

    return predicted_digit, confidence_score


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    image_path = "test.jpg"

    digit, confidence = predict_image(image_path)

    print("\n==========================")
    print(f"Predicted Digit : {digit}")
    print(f"Confidence      : {confidence:.2f}%")
    print("==========================")