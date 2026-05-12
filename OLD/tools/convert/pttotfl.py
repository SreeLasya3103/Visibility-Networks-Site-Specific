import torch
import torch.nn as nn
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np
import shutil
from torchvision import io
import sys
import os
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
import util.image_cropping
from models import *

INPUT_SHAPE = (1,3,280,280)
MODEL_NAME = "rmep_softmax"
MODEL_PATH = '/home/feet/Documents/SchoolResearch/VisibilityResearch/OLD/CurrentMobileModels/Testing/RMEP/RMEP-280.pt'
MODEL_MODULE = RMEP
OPTIMIZE = False
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

model = MODEL_MODULE.Model(10, 3, empty[0].clone(), empty[0].clone())
model.load_state_dict(torch.load(MODEL_PATH, weights_only=False, map_location=torch.device('cpu')))
model = WithSoftmax(model)
model.eval()
for m in model.modules():
    m.train(False)
onnx_filename = MODEL_NAME + ".onnx"

data = torch.randn(INPUT_SHAPE, dtype=torch.float32)
# data = io.read_image(IMG_PATH, io.ImageReadMode.RGB)/255.0
# data = resize_func(data)
# data = tf_function(data)
# data = torch.unsqueeze(data, 0)

preconv = model(data)
torch.onnx.export(model, data, MODEL_NAME, input_names=['input'], output_names=['output'], opset_version=11)

onnx_model = onnx.load(MODEL_NAME)
tf_rep = prepare(onnx_model)

tf_model_dir = "./tf_model"
tf_rep.export_graph(tf_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

converter.optimizations = []
if OPTIMIZE:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]


tflite_model = converter.convert()

with open(MODEL_NAME+ ".tflite", "wb") as f:
    f.write(tflite_model)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL_NAME+'.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

data = np.array(data, dtype=np.float32)

# Test the model on random input data.
input_data = np.array(data, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
postconv = interpreter.get_tensor(output_details[0]['index'])

os.remove(MODEL_NAME)
shutil.rmtree('tf_model')

print(preconv)
print(postconv)