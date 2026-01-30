"""
Custom Dataset Loader - Simplified for any fake/real video dataset
"""

import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDeepfakeDataset(Dataset):
    """
    Simplified dataset for custom deepfake data
    Works with any folder structure containing fake and real videos
    """
    
    def __init__(self, data_root, json_file, transform=None, num_frames=8,
                 fake_folder='fake_videos', real_folder='real_videos'):
        """
        Args:
            data_root: Root directory containing the dataset
            json_file: Path to JSON file with video IDs
            transform: Torchvision transforms to apply
            num_frames: Number of frames to sample per video
            fake_folder: Name of folder containing fake videos (relative to data_root)
            real_folder: Name of folder containing real videos (relative to data_root)
        """
        self.data_root = data_root
        self.num_frames = num_frames
        self.transform = transform
        self.fake_folder = fake_folder
        self.real_folder = real_folder
        
        # Load video IDs from JSON
        with open(json_file, 'r') as f:
            self.video_ids = json.load(f)
        
        # Build dataset list
        self.samples = []
        self._build_dataset()
        
        print(f"Loaded {len(self.samples)} samples from {json_file}")
    
    def _build_dataset(self):
        """Build list of video samples with their labels"""
        
        fake_dir = os.path.join(self.data_root, self.fake_folder)
        real_dir = os.path.join(self.data_root, self.real_folder)
        
        for video_id in self.video_ids:
            # Check if it's a fake video
            fake_path = os.path.join(fake_dir, str(video_id))
            if os.path.exists(fake_path):
                frames = sorted([f for f in os.listdir(fake_path) 
                               if f.endswith(('.png', '.jpg', '.jpeg'))])
                if frames:
                    self.samples.append({
                        'video_dir': fake_path,
                        'frames': frames,
                        'label': 1,  # Fake
                        'video_id': video_id
                    })
                    continue
            
            # Check if it's a real video
            real_path = os.path.join(real_dir, str(video_id))
            if os.path.exists(real_path):
                frames = sorted([f for f in os.listdir(real_path)
                               if f.endswith(('.png', '.jpg', '.jpeg'))])
                if frames:
                    self.samples.append({
                        'video_dir': real_path,
                        'frames': frames,
                        'label': 0,  # Real
                        'video_id': video_id
                    })
                    continue
            
            # Video not found
            print(f"Warning: Video {video_id} not found in fake or real folders")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_dir = sample['video_dir']
        frames = sample['frames']
        label = sample['label']
        
        # Sample frames uniformly
        if len(frames) <= self.num_frames:
            selected_frames = frames
        else:
            indices = torch.linspace(0, len(frames) - 1, self.num_frames).long()
            selected_frames = [frames[i] for i in indices]
        
        # Load and transform frames
        frame_tensors = []
        for frame_name in selected_frames:
            frame_path = os.path.join(video_dir, frame_name)
            try:
                img = Image.open(frame_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                frame_tensors.append(img)
            except Exception as e:
                print(f"Error loading {frame_path}: {e}")
                # Use a black frame as fallback
                if self.transform:
                    img = self.transform(Image.new('RGB', (224, 224)))
                else:
                    img = torch.zeros(3, 224, 224)
                frame_tensors.append(img)
        
        # Stack frames: [num_frames, C, H, W]
        frames_tensor = torch.stack(frame_tensors)
        
        # Average pooling over frames: [C, H, W]
        image = frames_tensor.mean(dim=0)
        
        return {
            'image': image,
            'label': label,
            'video_id': sample['video_id']
        }


def get_custom_dataloader(data_root, json_file, batch_size=16, num_workers=4,
                          num_frames=8, fake_folder='fake_videos', 
                          real_folder='real_videos', is_train=True):
    """
    Create dataloader for custom dataset
    
    Args:
        data_root: Root directory of dataset
        json_file: JSON file with video IDs
        batch_size: Batch size
        num_workers: Number of data loading workers
        num_frames: Frames to sample per video
        fake_folder: Folder name for fake videos
        real_folder: Folder name for real videos
        is_train: Whether this is training data (affects augmentation)
    
    Returns:
        DataLoader
    """
    
    # Define transforms
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    # Create dataset
    dataset = CustomDeepfakeDataset(
        data_root=data_root,
        json_file=json_file,
        transform=transform,
        num_frames=num_frames,
        fake_folder=fake_folder,
        real_folder=real_folder
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
    
    return dataloader
