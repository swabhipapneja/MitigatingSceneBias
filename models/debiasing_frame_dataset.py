import json
from PIL import Image
import os
import torch
from torch.utils.data import Dataset

class FrameDataset(Dataset):
    def __init__(self, frame_dir, mask_dir, scene_labels_path, transform, sample_duration=16, test_mode=False):
        """
        Args:
        - frame_dir: Path to the directory containing extracted frames.
        - mask_dir: Path to the directory containing masked frames.
        - scene_labels_path: Path to the JSON file containing pseudo scene labels.
        - transform: Transformations to apply to frames.
        - sample_duration: Number of frames to sample per video.
        """
        self.frame_dir = frame_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.sample_duration = sample_duration
        self.video_paths = []
        self.action_labels = []
        self.scene_labels = {}
        self.test_mode = test_mode

        # Loading pseudo scene labels
        with open(scene_labels_path, 'r') as f:
            self.scene_labels = json.load(f)

        # preparing test data
        if self.test_mode:
            # Test mode: Only video directories, no class_id directories
            for video_folder in os.listdir(frame_dir):
                video_path = os.path.join(frame_dir, video_folder)
                if os.path.isdir(video_path) and len(os.listdir(video_path)) > 0:
                    self.video_paths.append(video_path)
        
        else:
            # Collecting all video paths and corresponding class IDs
            for class_id in os.listdir(frame_dir):
                class_dir = os.path.join(frame_dir, class_id)
                for video_folder in os.listdir(class_dir):
                    video_path = os.path.join(class_dir, video_folder)
                    #if len(os.listdir(video_path)) > 0:
                    if len(os.listdir(video_path)) > 0:
                        self.video_paths.append(video_path)
                        self.action_labels.append(int(class_id))

        if not self.test_mode:
            assert len(self.video_paths) == len(self.action_labels), "Mismatch in video_paths and action_labels length."
            
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        if idx >= len(self.video_paths):
            raise IndexError(f"Index {idx} is out of range for video_paths with length {len(self.video_paths)}")

        video_path = self.video_paths[idx]
        action_label = self.action_labels[idx] if not self.test_mode else -1
        
        # One-hot encode the action label (only if not in test mode)
        if not self.test_mode:
            action_label_one_hot = torch.nn.functional.one_hot(
                torch.tensor(action_label, dtype=torch.long), 
                num_classes=200
            ).float()
        else:
            action_label_one_hot = torch.zeros(200, dtype=torch.float32)
            
        # if isinstance(action_label, int):
        #     action_label = torch.tensor(action_label, dtype=torch.long)

        # Load frames from the original video folder
        frame_files = sorted([os.path.join(video_path, img) for img in os.listdir(video_path)])
        if len(frame_files) == 0:
            raise ValueError(f"No frames found in video folder: {video_path}")

        if len(frame_files) > self.sample_duration:
            frame_files = frame_files[:self.sample_duration]
        elif len(frame_files) < self.sample_duration:
            padding = self.sample_duration - len(frame_files)
            frame_files += [frame_files[-1]] * padding

        frames = [self.transform(Image.open(f).convert('RGB')) for f in frame_files]
        video_tensor = torch.stack(frames, dim=1)

        # Loading masked frames
        mask_video_path = os.path.join(self.mask_dir, os.path.relpath(video_path, self.frame_dir))
        mask_frame_files = sorted([os.path.join(mask_video_path, img) for img in os.listdir(mask_video_path)])
        if len(mask_frame_files) == 0:
            raise ValueError(f"No masked frames found in video folder: {mask_video_path}")

        if len(mask_frame_files) > self.sample_duration:
            mask_frame_files = mask_frame_files[:self.sample_duration]
        elif len(mask_frame_files) < self.sample_duration:
            padding = self.sample_duration - len(mask_frame_files)
            mask_frame_files += [mask_frame_files[-1]] * padding

        mask_frames = [self.transform(Image.open(f).convert('RGB')) for f in mask_frame_files]
        mask_video_tensor = torch.stack(mask_frames, dim=1)
        
        video_key = os.path.relpath(video_path, self.frame_dir)
        scene_label = self.scene_labels.get(video_key, -1)  # Default to -1 if not found
        if scene_label == -1:
            scene_label_one_hot = torch.zeros(365, dtype=torch.float32)  # A zero vector for invalid labels
        else:
            # One-hot encode the scene label
            scene_label_one_hot = torch.nn.functional.one_hot(torch.tensor(scene_label, dtype=torch.long), num_classes=365).float()


        if self.test_mode:
            return video_tensor, mask_video_tensor, scene_label_one_hot
        else:
            return video_tensor, mask_video_tensor, action_label_one_hot, scene_label_one_hot

