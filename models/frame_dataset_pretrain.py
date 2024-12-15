from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class FrameDataset(Dataset):
    def __init__(self, frame_dir, transform=None, sample_duration=25):
        self.frame_dir = frame_dir
        self.transform = transform
        self.sample_duration = sample_duration
        self.video_paths = []
        self.labels = []

        for class_id in os.listdir(frame_dir):
            class_dir = os.path.join(frame_dir, class_id)
            if not os.path.isdir(class_dir):  # Skip non-directory entries
                continue
            video_folders = os.listdir(class_dir)

            # Skipping classes with fewer than 2 samples
            if len(video_folders) < 2:
                continue

            for video_folder in video_folders:
                video_path = os.path.join(class_dir, video_folder)
                if os.path.isdir(video_path) and len(os.listdir(video_path)) > 0:  
                    self.video_paths.append(video_path)
                    self.labels.append(int(class_id))

        assert len(self.video_paths) == len(self.labels), "Mismatch in video_paths and labels length."

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        if idx >= len(self.video_paths):
            raise IndexError(f"Index {idx} is out of range for video_paths with length {len(self.video_paths)}")

        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Loading frames from the video folder
        frame_files = sorted([
            os.path.join(video_path, img)
            for img in os.listdir(video_path)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))  # Filter for image files
        ])
        
        if len(frame_files) == 0:
            raise ValueError(f"No valid frames found in video folder: {video_path}")
        
        # Selecting frames based on sample_duration
        if len(frame_files) > self.sample_duration:
            frame_files = frame_files[:self.sample_duration]
        elif len(frame_files) < self.sample_duration:
            padding = self.sample_duration - len(frame_files)
            frame_files += [frame_files[-1]] * padding

        frames = []
        for frame_file in frame_files:
            try:
                frame = Image.open(frame_file).convert('RGB')  # Load valid images
            except Exception as e:
                raise ValueError(f"Error loading image: {frame_file}. Error: {e}")
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        video = torch.stack(frames, dim=1)
        return video, label


# Function to create train, val, and test DataLoaders
def create_dataloaders(frame_dir, transform=None, batch_size=8, train_ratio=0.7, val_ratio=0.15, random_seed=42):
    # Initialize the base dataset
    dataset = FrameDataset(frame_dir, transform=transform)

    # Splitting indices for train, val, and test
    indices = list(range(len(dataset)))
    labels = dataset.labels  # Extract labels for stratified splitting

    train_idx, temp_idx = train_test_split(
        indices, test_size=(1 - train_ratio), random_state=random_seed
    )
    
    temp_labels = [labels[i] for i in temp_idx]
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_ratio_adjusted), random_state=random_seed
    )

    # Creating Subset datasets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
