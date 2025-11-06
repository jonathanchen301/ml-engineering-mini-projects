import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

transforms_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

transforms_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

class XRayDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.transforms = transforms
        self.samples = []  # Precompute list of (filepath, label)

        # Build list once
        for label, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
            class_dir = os.path.join(path, class_name)
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, filename), label))

        print(f"Normal Samples: {len([s for s in self.samples if s[1] == 0])}")
        print(f"Pneumonia Samples: {len([s for s in self.samples if s[1] == 1])}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, label

def create_datasets(path):
    train_dataset = XRayDataset(path=os.path.join(path, "train"), transforms=transforms_train)
    val_dataset = XRayDataset(path=os.path.join(path, "test"), transforms=transforms_val)
    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset, batch_size, num_workers=0):
    if torch.cuda.is_available():
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        # For some reason, num_workers > 0 makes it crash.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


