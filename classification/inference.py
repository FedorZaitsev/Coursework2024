import sys
import os
import torch
import torchvision.transforms as T
from PIL import Image

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from util import get_model


class Model():
    def __init__(self, model_type, model_state_dict):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if model_type == 'QResNet18':
            self.device = torch.device('cpu')
        self.model = get_model(model_type)()
        self.model.load_state_dict(torch.load(model_state_dict, map_location=self.device))

    def inference(self, filename, transform=None):
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
        
        image = image.unsqueeze(0)
        self.model = self.model.to(self.device)
        image = image.to(self.device)
        self.model.eval()
        return self.model(image)

if __name__ == "__main__":
    model_state = '/home/fedor/Coursework2024/model2'
    model = Model('ConvModel', model_state)

    img = '/home/fedor/Coursework2024/example/1.jpg'

    print(model.inference(img))
