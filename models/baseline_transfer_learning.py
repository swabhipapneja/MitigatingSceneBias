from frame_dataset_baseline import FrameDataset
import torch
from torchvision.models.video import r3d_18
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, Resize, CenterCrop, Lambda, Normalize, ToTensor
import torch
from torchvision.models.video import r3d_18

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2):
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=(1 - train_ratio), random_state=42)
    return train_idx, val_idx


# Feature Extractor Model (BaseNet)
class FeatureExtractor(nn.Module):
    def __init__(self, num_classes=101):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),  # Example layer
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        output = self.fc(features)
        return output

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, num_epochs=1):
    for epoch in range(num_epochs):
        model = model.cuda()  # or model.to('cuda')

        model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        train_accuracy = 100 * correct / total

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        val_accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}/{num_epochs} -> "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_accuracy:.2f}%")

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 30
    
    # Defining the preprocessing steps
    transform = Compose([
        Resize((112, 112)),  # Resizing frames to a consistent size
        CenterCrop(112),  # Cropping the center of each frame
        ToTensor(),  # Converting images to PyTorch tensors
        Lambda(lambda x: x / 255.0),  # Scale pixel values to [0, 1]
        Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])  # Normalize
    ])

    train_frame_dir = '/scratch/supalami/ProjectDataSplitwise/lastminutedata/extractionoutput/train'
    test_frame_dir = '/scratch/supalami/ProjectDataSplitwise/lastminutedata/extractionoutput/test'
    
    train_dataset = FrameDataset(frame_dir=train_frame_dir, transform=transform, sample_duration=16)
    test_dataset = FrameDataset(frame_dir=train_frame_dir, transform=transform, sample_duration=16)
    
    train_idx, val_idx = split_dataset(train_dataset)
    
    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(train_dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    import torch
    from torchvision.models.video import r3d_18

    model = r3d_18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 48)  # Update 48 for Diving48, 101 for UCf101 and 51 for HMDB51 dataset
    # Loading the pretrained weights
    pretrained_weights = torch.load("models_saved/baseline_feature_extractor.pth")
    # Modifying keys to match the current model
    modified_weights = {key.replace("feature_extractor.", ""): value for key, value in pretrained_weights.items()}
    # Loading the modified state_dict
    model.load_state_dict(modified_weights, strict=False)

    # Optimizer and Criterion
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    model = model.cuda()  # or model.to('cuda')

    train_and_validate(model, train_loader, val_loader, optimizer, criterion, num_epochs)

    torch.save(model.state_dict(), "baseline_diving48_trained_model.pth")
    print("Model saved to baseline_diving48_trained_model.pth")



    
    