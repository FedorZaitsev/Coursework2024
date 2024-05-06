import sys
import os
import torch
import torchvision.transforms as T
import albumentations as A
from PIL import Image

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from util import get_model, parse_config


class Model():
    def __init__(self, model_type, model_state_dict, inference_info):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.inference_info = inference_info
        if model_type == 'QuantizedResNet18':
            self.device = torch.device('cpu')
        self.model = get_model(model_type)(num_classes=len(self.inference_info['classes']))
        self.model.load_state_dict(torch.load(model_state_dict, map_location=self.device))

    def inference(self, filename):
        x_t = T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
            ]
        )
        image = Image.open(filename)
        image = x_t(image)
        
        transform = None
        if len(self.inference_info['valid_transforms']) > 0:
            transform = A.from_dict(self.inference_info['valid_transforms'])

        if transform:
            transformed = transform(image=image.permute(1, 2, 0).numpy())
            image = torch.tensor(transformed['image']).permute(2, 0, 1)
        
        image = image.unsqueeze(0)
        self.model = self.model.to(self.device)
        image = image.to(self.device)
        self.model.eval()
        output = self.model(image)
        _, y_pred = torch.max(output, 1)
        # print(output)
        return self.inference_info['classes'][y_pred]

if __name__ == "__main__":
    model_state = '../models/ResNet18_model'
    inference_info = {"classes": ["Ill_cucumber", "good_Cucumber"], "valid_transforms": {}}
    model = Model('ResNet18', model_state, inference_info)

    img = '/home/fedor/Coursework2024/example/5.jpg'

    print(model.inference(img))
