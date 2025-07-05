from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ProcessedToCleanDataset(Dataset):
    def __init__(self, processed_dir, target_dir, transform=None):
        self.transform = transform
        self.processed_images = [f for f in os.listdir(processed_dir) if f.endswith('.jpg')]
        self.target_images = [f for f in os.listdir(target_dir) if f.endswith('.jpg')]
        self.processed_dir = processed_dir
        self.target_dir = target_dir
        self.images = list(set(self.processed_images).intersection(self.target_images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        proc_img = Image.open(os.path.join(self.processed_dir, filename)).convert('RGB')
        targ_img = Image.open(os.path.join(self.target_dir, filename)).convert('RGB')
        if self.transform:
            proc_img = self.transform(proc_img)
            targ_img = self.transform(targ_img)
        return proc_img, targ_img
