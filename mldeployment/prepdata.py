#from pathlib import Path
from typing import Tuple
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


NUM_CLASSES = 4
INPUT_SHAPE = (128, 128, 3)
NUM_EPOCHS = 1
image_dir = "/home/christian/Desktop/Programs/Python/AI/Deployment/BentoML/mldeployment/images/"
BATCH_SIZE = 4



def prepare_data() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(size=(128, 128)),
        transforms.Normalize((0.5,), (0.5,))])



    # Define a custom dataset class
    class CustomImageDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.images = os.listdir(root_dir)

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_name = os.path.join(self.root_dir, self.images[idx])
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            return image

    # Create custom dataset and dataloader
    custom_dataset = CustomImageDataset(root_dir=image_dir, transform=data_transform)
    dataloader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return dataloader
