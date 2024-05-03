import torch.nn as nn
import torch

def ConvModel():
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, padding='same', kernel_size=9),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.45),    

        nn.Conv2d(in_channels=32, out_channels=32, padding='same', kernel_size=9),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.45),    

        nn.Conv2d(in_channels=32, out_channels=32, padding='same', kernel_size=9),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.45),    

        nn.Conv2d(in_channels=32, out_channels=32, padding='same', kernel_size=9),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.45),    
        
        nn.Flatten(),
        nn.Linear(16 * 16 * 32, 1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 2)
    )

    return model

def Swin(size='t', pretrained=True):

    model = None
    weights = None

    if size == 't':
        from torchvision.models import swin_v2_t,  Swin_V2_T_Weights

        if pretrained:
            weights = Swin_V2_T_Weights.IMAGENET1K_V1

        model = nn.Sequential(
            swin_v2_t(num_classes=1000, weights=weights),
            nn.Linear(1000, 2)
        )

    if size == 's':
        from torchvision.models import swin_v2_s,  Swin_V2_S_Weights

        if pretrained:
            weights = Swin_V2_S_Weights.IMAGENET1K_V1

        model = nn.Sequential(
            swin_v2_s(num_classes=1000, weights=weights),
            nn.Linear(1000, 2)           
        )

    if size == 'b':
        from torchvision.models import swin_v2_b,  Swin_V2_B_Weights
        
        if pretrained:
            weights = Swin_V2_B_Weights.IMAGENET1K_V1
        
        model = nn.Sequential(
            swin_v2_b(num_classes=1000, weights=weights),
            nn.Linear(1000, 2)            
        )

    return model

def QuantizableResNet18(pretrained=True):
    from torchvision.models.quantization import resnet18
    weights=None
    if pretrained:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1

    model = nn.Sequential(
        resnet18(num_classes=1000, weights=weights),
        torch.ao.quantization.QuantStub(),
        nn.Linear(1000, 2)
    )

    return model

def QuantizedResNet18():
    model_fp32 = QuantizableResNet18(pretrained=False)
    model_fp32.eval()
    model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
    model_fp32_prepared = torch.quantization.prepare(model_fp32, inplace = False).to('cpu')
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    return model_int8
