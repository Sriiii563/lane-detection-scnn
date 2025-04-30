import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import config

class CULaneDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples(split_file)

    def _load_samples(self, split_file):
        with open(os.path.join(self.root_dir, split_file), 'r') as f:
            lines = f.readlines()
        samples = []
        for line in lines:
            line = line.strip()
            img_path = os.path.join(self.root_dir, line)
            mask_path = img_path.replace('driver_', 'laneseg_label_w16/driver_').replace('.jpg', '.png')
            samples.append((img_path, mask_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask).long().squeeze(0)  # Convert mask to tensor and remove channel

        return image, mask
