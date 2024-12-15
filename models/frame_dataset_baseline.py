from PIL import Image
from torch.utils.data import Dataset
import os
import torch

class FrameDataset(Dataset):
    def __init__(self, frame_dir, transform=None, sample_duration=25):
        self.frame_dir = frame_dir
        self.transform = transform
        self.sample_duration = sample_duration
        self.video_paths = []
        self.labels = []
        self.class_to_idx = {} 

        # Collecting all video paths and corresponding class labels
        class_names = sorted(os.listdir(frame_dir))  # Sorting for consistent label ordering
        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx  # Mapping class name to an integer label
            class_dir = os.path.join(frame_dir, class_name)
            for video_folder in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_folder)
                if len(os.listdir(video_path)) > 0:  
                    self.video_paths.append(video_path)
                    self.labels.append(self.class_to_idx[class_name])  # Using mapped integer label

        assert len(self.video_paths) == len(self.labels), "Mismatch in video_paths and labels length."


    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        if idx >= len(self.video_paths):
            raise IndexError(f"Index {idx} is out of range for video_paths with length {len(self.video_paths)}")

        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Debugging: Check paths and index
        #print(f"Accessing index: {idx}, video path: {video_path}")

        # Loading frames from the video folder
        frame_files = sorted([os.path.join(video_path, img) for img in os.listdir(video_path)])
        
        if len(frame_files) == 0:
            raise ValueError(f"No frames found in video folder: {video_path}")
        
        # Selecting frames based on sample_duration
        if len(frame_files) > self.sample_duration:
            frame_files = frame_files[:self.sample_duration]
        elif len(frame_files) < self.sample_duration:
            padding = self.sample_duration - len(frame_files)
            frame_files += [frame_files[-1]] * padding

        frames = []
        for frame_file in frame_files:
            frame = Image.open(frame_file).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        video = torch.stack(frames, dim=1)
        return video, label
