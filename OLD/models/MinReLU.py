import torch.nn as nn
import torchvision.transforms as tf

class Model(nn.Module):
    def __init__(self, num_classes, num_channels, mean, std):
        super(Model, self).__init__()

        if mean is None or std is None:
            self.normalize = nn.Identity()
        else:
            self.normalize = tf.Normalize(mean, std)

        model = [nn.Conv2d(num_channels, 16, 3),
                 nn.MaxPool2d(2,2)]
        
        model += [nn.Flatten(),
                  nn.LazyLinear(num_classes),
                  nn.ReLU()]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        x = self.normalize(x)
        
        return self.model(x)
    
def get_tf_function():
    def transform(img):
        return img
    
    return transform