#This script is for checking the number of images in each class of a dataset with option of limits on classes.

import torch
from torch.utils.data import DataLoader
import sys
import pathlib
from .. import dsets

DSET_DIR = '/home/feet/Documents/SchoolResearch/VisibilityResearch/Datasets/datasets/NewWebcams/'
CLASS_NAMES = ('1.0', '1.25', '1.5', '1.75', '2.0', '2.25', '2.5', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0')
# CLASS_NAMES = ('1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0')
LIMITS = {3.0: 800, 4.0: 800, 5.0: 800, 6.0: 800, 7.0: 800, 8.0: 800, 9.0: 800, 10.0: 800}

dataset = dsets.Webcams.Webcams_cls(DSET_DIR, transformer=lambda x:x, limits=LIMITS)
dataloader = DataLoader(dataset, 1, False)

class_counts = dict()

for i, (data, label, _) in enumerate(dataloader):
    argmax = torch.argmax(label).item()
    if argmax in class_counts:
        class_counts[argmax] += 1
    else:
        class_counts[argmax] = 1

_, counts_list = zip(*sorted(class_counts.items(), key=lambda x: x[0]))

counts_list = list(zip(CLASS_NAMES, counts_list))

for name, count in counts_list:
    print(count, end=', ')
print('')