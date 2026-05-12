#This script generates a collage of images from a datset.
import os
import torch.utils
import torch.utils.data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torchvision as tv
import torch
import sys
import pathlib
pathlib.Path(__file__).parent.resolve()
sys.path.append(pathlib.Path(__file__).parent.resolve())
from util import image_cropping
from dsets.Webcams import Webcams_cls_10

LIMITS = {1.75:232, 2.0:232, 3.0:521, 4.0:521, 5.0:521, 6.0:521, 7.0:521, 8.0:521, 9.0:521, 10.0:521}

w = 5
h = 5
which = 0

dsets = []

dsets.append(Webcams_cls_10('/home/feet/Documents/LAWN/datasets/quality-labeled-webcams/by-network/good', limits=LIMITS, transformer=image_cropping.get_resize_crop_fn((310,470))))

loaded_images = []

dl = torch.utils.data.DataLoader(dsets[0], 1, True, num_workers=4)
i = 0
for img, _, _ in dl:
    loaded_images.append(img[0])
    if i > 50:
        break
    i += 1
    
    
final_image = None

for y in range(h):
    row = loaded_images[y*w]
    for x in range(1, w):
        next_img = loaded_images[y*w + x]
        row = torch.cat((row, next_img), 2)
    
    if y == 0:
        final_image = row
    else:
        final_image = torch.cat((final_image, row), 1)

tv.utils.save_image(final_image, "./random_images.jpg")