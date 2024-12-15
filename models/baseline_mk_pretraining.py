from frame_dataset_pretrain import FrameDataset
from frame_dataset_pretrain import create_dataloaders
import torch
from torchvision.models.video import r3d_18
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, Lambda, Normalize, ToTensor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import os


preprocess = Compose([
    Resize((112, 112)),  # Resize frames to a consistent size
    CenterCrop(112),  # Crop the center of each frame
    ToTensor(),  # Convert images to PyTorch tensors
    Lambda(lambda x: x / 255.0),  # Scale pixel values to [0, 1]
    Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])  # Normalize
])

frame_dir = '../ProjectDataSplitwise/extracted_frames'  # Update this path to your dataset
train_loader, val_loader, test_loader = create_dataloaders(
    frame_dir, transform=preprocess, batch_size=8, train_ratio=0.7, val_ratio=0.15
)

print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

for videos, labels in train_loader:
    print(videos.shape, labels.shape)  # Expected: (batch_size, C, T, H, W), (batch_size,)
    break


from torchvision.models.video import r3d_18
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Load the ResNet-3D backbone without the final fully connected layer
        resnet3d = r3d_18(pretrained=False)
        # children() gives a list of all the layers in the ResNet-3D model
        # [:-2] removes the last two layers
        # the fully connected (FC) layer that outputs class predictions.
        # the global average pooling layer, which compresses spatial and temporal dimensions.
        # The code removes the final layers (used for classification)
        self.feature_extractor = nn.Sequential(*list(resnet3d.children())[:-2])  # Remove FC and AvgPool
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Add adaptive pooling for feature compression

    def forward(self, x):
        x = self.feature_extractor(x)  # Extract spatiotemporal features
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten for classifier input
        return x

class ActionClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(ActionClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)



feature_extractor = FeatureExtractor()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_extractor = feature_extractor.to(device)

action_classifier = ActionClassifier(feature_dim=512, num_classes=200)  # Adjust feature_dim based on extractor output
action_classifier = action_classifier.to(device)

print(feature_extractor)
print(action_classifier)

criterion = nn.CrossEntropyLoss()

# Definining optimizer for your model parameters
optimizer = optim.SGD(
    list(feature_extractor.parameters()) + list(action_classifier.parameters()),  # Model parameters to optimize
    lr=0.0001,  # Learning rate
    momentum=0.9,  # Momentum to stabilize training
    weight_decay=1e-4  # Weight decay for regularization (optional)
)


def evaluate_model(feature_extractor, action_classifier, data_loader, device):
    """Evaluate the model on a given dataset and return accuracy."""
    feature_extractor.eval()  # Set to evaluation mode
    action_classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for videos, labels in data_loader:
            videos, labels = videos.to(device), labels.to(device)

            features = feature_extractor(videos)
            features = features.to(device)
            outputs = action_classifier(features)

            _, predicted = torch.max(outputs, 1)  # Index of the max logit

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
best_val_accuracy = 0.0  # 


num_epochs = 10
validation_interval = 2  

for epoch in range(num_epochs):
    feature_extractor.train()
    action_classifier.train()

    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        features = feature_extractor(videos)
        features = features.to(device)
        outputs = action_classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * videos.size(0)

        _, predicted = torch.max(outputs, 1)  
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = 100 * correct_train / total_train
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")
    if (epoch + 1) % validation_interval == 0:
        val_accuracy = evaluate_model(feature_extractor, action_classifier, val_loader, device)
        print(f"Validation Accuracy after Epoch {epoch+1}: {val_accuracy:.2f}%")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"New best validation accuracy: {best_val_accuracy:.2f}%. Saving model.")
            torch.save(feature_extractor.state_dict(), 'baseline_feature_extractor.pth')
            torch.save(action_classifier.state_dict(), 'baseline_action_classifier.pth')

test_accuracy = evaluate_model(feature_extractor, action_classifier, test_loader, device)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")


