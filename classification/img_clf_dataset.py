import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
from tqdm import tqdm


class ClassDataset(Dataset):
    def __init__(self, root_dir, idx, transform=None, pred_mode=None):
                
        self.transform = transform
        self.pred_mode = pred_mode    
        self.root_dir = root_dir
        _, dirs, _ = next(os.walk(self.root_dir))
        self.classes = sorted(dirs)
        
        self.labels = []
        self.files = []
        for i, c in enumerate(self.classes):    
            p, _, filenames = next(os.walk(os.path.join(self.root_dir, c)))
            filenames = sorted(list(map(lambda f: os.path.join(p, f), filenames)))
            self.files += filenames
            self.labels += [i] * len(filenames)
        
        self.labels = torch.Tensor(self.labels)[idx]
        self.data = torch.empty(size=(len(idx), 3, 256, 256))
        x_t = T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
            ]
        )
        for i, filename in tqdm(enumerate(map(self.files.__getitem__, idx)), desc='Loading data ...'):                
            image = Image.open(filename)
            image = x_t(image)
            
            if self.transform:
                transformed = transform(image=image.permute(1, 2, 0).numpy())
                image = torch.tensor(transformed['image']).permute(2, 0, 1)
            
            self.data[i] = image 
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label = self.labels[idx]
        image = self.data[idx]
        return image, label


def create_dataloader(cur_dir, train_ratio=0.8, train_transforms=[], valid_transforms=[], 
                      batch_size=32, num_workers=4, use_original=True):
    
    data_size = 0
    p, classes_train, _ = next(os.walk(cur_dir))
    for c in classes_train:
        _, _, f = next(os.walk(os.path.join(p, c)))
        data_size += len(f)

    train_size = int(train_ratio * data_size)
    valid_size = data_size - train_size

    idx_train, idx_valid = torch.utils.data.random_split(list(range(data_size)), [train_size, valid_size])
    idx_train = list(idx_train)
    idx_valid = list(idx_valid)

    train_dataset_aug = [None] * len(train_transforms)
    valid_dataset_aug = [None] * len(valid_transforms)

    for i, transform in enumerate(train_transforms):
        train_dataset_aug[i] = ClassDataset(root_dir=cur_dir, 
                                         idx=idx_train, transform=transform)

    for i, transform in enumerate(valid_transforms):
        valid_dataset_aug[i] = ClassDataset(root_dir=cur_dir, 
                                         idx=idx_valid, transform=transform)

    if use_original:
        
        train_dataset = ClassDataset(root_dir=cur_dir,
                             idx=idx_train, transform=None)
        valid_dataset = ClassDataset(root_dir=cur_dir,
                             idx=idx_valid, transform=None)
        
        train_dataset_aug += [train_dataset]
        valid_dataset_aug += [valid_dataset]
    
    train_loader = DataLoader(ConcatDataset(train_dataset_aug), 
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                              pin_memory=True)
    valid_loader = DataLoader(ConcatDataset(valid_dataset_aug), 
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                              pin_memory=True)
    
    return train_loader, valid_loader