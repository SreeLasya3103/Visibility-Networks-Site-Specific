import ai_edge_torch
import numpy
import torch
import torch.nn as nn
from torchvision import io

import sys
import os
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
import util.image_cropping
from models import *

INPUT_SHAPE = (1,3,3,280,280)
N_CLASSES = 10
MODEL_NAME = "VisNet_finetuned"
MODEL_PATH = '/home/feet/Documents/SchoolResearch/VisibilityResearch/finetuning/VisNet/runs/Jul29_10-25-18_node008/best-loss.pt'
MODEL_MODULE = VisNet
# OPTIMIZE = False
IMG_PATH = '/home/feet/Documents/SchoolResearch/VisibilityResearch/Datasets/datasets/NewWebcams/4-9-24-6-47-PM/SITE7_ORNT312_VIS9mi.png'

empty = torch.zeros(INPUT_SHAPE)
tf_function = MODEL_MODULE.get_tf_function()
resize_func = util.image_cropping.get_resize_crop_fn(INPUT_SHAPE[-2:])

class WithSigmoid(nn.Module):
    def __init__(self, model):
        super(WithSigmoid, self).__init__()

        self.model = model
    
    def forward(self, x):
        return nn.functional.sigmoid(self.model(x))
    
class WithSoftmax(nn.Module):
    def __init__(self, model):
        super(WithSoftmax, self).__init__()

        self.model = model

    def forward(self, x):
        return nn.functional.softmax(self.model(x), 1)

model = MODEL_MODULE.Model(N_CLASSES, 3, empty[0].clone(), empty[0].clone())
model.load_state_dict(torch.load(MODEL_PATH, weights_only=False, map_location=torch.device('cpu')))
model = WithSoftmax(model)
model.eval()

data = io.decode_image(IMG_PATH, io.ImageReadMode.RGB)/255.0
data = resize_func(data)
data = tf_function(data)
data = torch.unsqueeze(data, 0)
data = (data,)

preconv = model(*data)

tfl_model = ai_edge_torch.convert(model.eval(), data)

postconv = tfl_model(*data)

print(preconv)
print(postconv)
if (numpy.allclose(
    preconv.detach().numpy(),
    postconv,
    atol=1e-5,
    rtol=1e-5,
)):
    print("Inference result with Pytorch and TfLite was within tolerance")
else:
    print("Something wrong with Pytorch --> TfLite")

tfl_model.export(f"./{MODEL_NAME}.tflite")