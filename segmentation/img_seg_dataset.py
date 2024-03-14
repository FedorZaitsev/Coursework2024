import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T

from tqdm import tqdm


class SegmentDataset(Dataset):
    def __init__(self, root_dir, root_dir_masks, idx, transform=None, pred_mode=None, m_mode=False):
                
        self.root_dir = root_dir
        self.root_dir_masks = root_dir_masks
        self.transform = transform
        self.pred_mode = pred_mode
        self.m_mode = m_mode
        
        _, _, files = next(os.walk(self.root_dir))
        files = sorted(files)
        
        self.data = torch.empty(size=(len(idx), 3, 256, 256))
        self.data_masks = torch.empty(size=(len(idx), 1, 256, 256))
        x_t = T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
            ]
        )
        
        y_t = T.Compose(
            [
                T.Resize((256, 256)),
                T.PILToTensor(),
                T.Lambda(lambda x: (x != 0).long())
            ]
        )
        for i, filename in tqdm(enumerate(map(files.__getitem__, idx)), desc='Loading data ...'):
            img_name = os.path.join(self.root_dir, filename)
            mask_name = os.path.join(self.root_dir_masks, filename[:-3] + 'png')
    
            
            image = Image.open(img_name).convert('RGB')
            image = x_t(image)
                
            mask = Image.open(mask_name)
            if self.m_mode:
                mask = mask.convert('L')
            mask = y_t(mask)
             
            if self.transform:
                transformed = transform(image=image.permute(1, 2, 0).numpy(), mask=mask.permute(1, 2, 0).numpy())
                image = torch.tensor(transformed['image']).permute(2, 0, 1)
                mask = torch.tensor(transformed['mask']).permute(2, 0, 1)
            
            self.data[i] = image 
            self.data_masks[i] = mask

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.data[idx]
        mask = self.data_masks[idx]
        return image, mask
    

def create_dataloader(cur_dir, train_ratio=0.8, train_transforms=[], valid_transforms=[], 
                      batch_size=32, num_workers=4, use_original=True):
    
    _, _, files = next(os.walk(cur_dir + 'image'))
    sample_size = len(files)
    train_size = int(train_ratio * sample_size)
    valid_size = sample_size - train_size

    idx_train, idx_valid = torch.utils.data.random_split(list(range(sample_size)), [train_size, valid_size])
    idx_train = list(idx_train)
    idx_valid = list(idx_valid)

    train_dataset_aug = [None] * len(train_transforms)
    valid_dataset_aug = [None] * len(valid_transforms)

    for i, transform in enumerate(train_transforms):
        train_dataset_aug[i] = SegmentDataset(root_dir=(cur_dir + 'image/'),
                                         root_dir_masks=(cur_dir + 'label/'), 
                                         idx=idx_train, transform=transform, m_mode=True)

    for i, transform in enumerate(valid_transforms):
        valid_dataset_aug[i] = SegmentDataset(root_dir=(cur_dir + 'image/'), 
                                         root_dir_masks=(cur_dir + 'label/'), 
                                         idx=idx_valid, transform=transform, m_mode=True)

    if use_original:
            
        train_dataset = SegmentDataset(root_dir=(cur_dir + 'image/'), root_dir_masks=(cur_dir + 'label/'),
                            idx=idx_train, transform=None, m_mode=True)

        valid_dataset = SegmentDataset(root_dir=(cur_dir + 'image/'), root_dir_masks=(cur_dir + 'label/'),
                            idx=idx_valid, transform=None, m_mode=True)
        
        train_dataset_aug += [train_dataset]
        valid_dataset_aug += [valid_dataset]
    
    train_loader = DataLoader(ConcatDataset(train_dataset_aug), 
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                              pin_memory=True)
    valid_loader = DataLoader(ConcatDataset(valid_dataset_aug), 
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                              pin_memory=True)
    
    return train_loader, valid_loader