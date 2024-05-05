import sys
import os
import albumentations as A
import torch
import gc
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


from classification.util import get_model
from classification.img_clf_train_pipeline import clf_train
from classification.img_clf_dataset import get_loaders
from classification.quantization import static_quantize

def inference(model, filename, transform=None):
    x_t = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
        ]
    )
    image = Image.open(filename)
    image = x_t(image)
    
    if transform:
        transformed = transform(image=image.permute(1, 2, 0).numpy())
        image = torch.tensor(transformed['image']).permute(2, 0, 1)
    
    
    
