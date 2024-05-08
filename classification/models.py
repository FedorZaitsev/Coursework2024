import torch.nn as nn
import torch

def ConvModel(num_classes=2, p=0):
    model = nn.Sequential(
        
        nn.Conv2d(in_channels=3, out_channels=16, padding='same', kernel_size=3),
        nn.BatchNorm2d(16),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p),    

        nn.Conv2d(in_channels=16, out_channels=32, padding='same', kernel_size=3),
        nn.BatchNorm2d(32),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p),    

        nn.Conv2d(in_channels=32, out_channels=64, padding='same', kernel_size=3),
        nn.BatchNorm2d(64),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p),    

        nn.Conv2d(in_channels=64, out_channels=128, padding='same', kernel_size=3),
        nn.BatchNorm2d(128),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p),    
        
        nn.Conv2d(in_channels=128, out_channels=256, padding='same', kernel_size=3),
        nn.BatchNorm2d(256),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(p),    
        
        nn.Flatten(),
        nn.Linear(8 * 8 * 256, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Linear(512, num_classes)

    )

    return model

def Swin(size='t', pretrained=False, num_classes=2):

    model = None
    weights = None

    if size == 't':
        from torchvision.models import swin_v2_t,  Swin_V2_T_Weights

        if pretrained:
            weights = Swin_V2_T_Weights.IMAGENET1K_V1

        model = nn.Sequential(
            swin_v2_t(num_classes=1000, weights=weights),
            nn.Linear(1000, num_classes)
        )

    if size == 's':
        from torchvision.models import swin_v2_s,  Swin_V2_S_Weights

        if pretrained:
            weights = Swin_V2_S_Weights.IMAGENET1K_V1

        model = nn.Sequential(
            swin_v2_s(num_classes=1000, weights=weights),
            nn.Linear(1000, num_classes)           
        )

    if size == 'b':
        from torchvision.models import swin_v2_b,  Swin_V2_B_Weights
        
        if pretrained:
            weights = Swin_V2_B_Weights.IMAGENET1K_V1
        
        model = nn.Sequential(
            swin_v2_b(num_classes=1000, weights=weights),
            nn.Linear(1000, num_classes)            
        )

    return model

def QuantizableResNet18(pretrained=False, num_classes=2):
    from torchvision.models.quantization import resnet18
    weights=None
    if pretrained:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1

    model = nn.Sequential(
        resnet18(num_classes=1000, weights=weights),
        torch.ao.quantization.QuantStub(),
        nn.Linear(1000, num_classes)
    )

    return model

def QuantizedResNet18(num_classes=2):
    model_fp32 = QuantizableResNet18(pretrained=False, num_classes=num_classes)
    model_fp32.eval()
    model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
    model_fp32_prepared = torch.quantization.prepare(model_fp32, inplace = False).to('cpu')
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    return model_int8
