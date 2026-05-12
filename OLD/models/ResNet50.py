import torch.nn as nn
import torchvision as tv

class Model(nn.Module):
    def __init__(self, num_classes, num_channels, mean, std):
        super(Model, self).__init__()
        
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        self.model = tv.models.resnet50(num_classes=num_classes)

    def forward(self, x):
        x = (x - self.mean) / self.std
        
        return self.model(x)


def get_tf_function():
    def transform(img):
        return img
    
    return transform