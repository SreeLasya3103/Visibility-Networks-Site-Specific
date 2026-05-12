import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision as tv
import torchvision.io as io
from torcheval.metrics import BinaryAccuracy
from glob import glob
import os
from os import path
from pathlib import Path
from random import Random
from math import ceil
import torchvision.transforms.functional as tff
import torchvision.transforms as tf
import onnx
from onnx_tf.backend import prepare
import numpy as np
import shutil
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.resolve()))
import util.image_cropping
from models import *

DSET_DIR = '/home/feet/Documents/SchoolResearch/VisibilityResearch/Datasets/datasets/NewWebcams'
MODEL_PATH = '/home/feet/Documents/SchoolResearch/VisibilityResearch/Visibility-Networks/rewrite2/tools/goodbad/goodbad.pt'
LISTS_PATH = '/home/feet/Documents/SchoolResearch/VisibilityResearch/Datasets/datasets/WebcamsGoodBadLists'

USE_CUDA = True
EPOCHS = 100
BATCH_SIZE = 16
LR = 0.000001
IMG_RES = (280,280)
MAX_GOOD = 406
MAX_BAD = 406

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class GoodBadWebcams(Dataset):
    def __init__(self, max_good, max_bad):
        with open(os.path.join(LISTS_PATH, 'good.csv')) as f:
            good_images = f.read().splitlines()
        with open(os.path.join(LISTS_PATH, 'bad.csv')) as f:
            bad_images = f.read().splitlines()

        for i in range(len(good_images)):
            good_images[i] = os.path.join(DSET_DIR, good_images[i])
        for i in range(len(bad_images)):
            bad_images[i] = os.path.join(DSET_DIR, bad_images[i])

        Random(37).shuffle(good_images)
        Random(37).shuffle(bad_images)

        if len(good_images) > max_good:
            good_images = good_images[:max_good]
        if len(bad_images) > max_bad:
            bad_images = bad_images[:max_bad]
        
        self.img_label_pairs = [(img, 1.0) for img in good_images]
        self.img_label_pairs += [(img, 0.0) for img in bad_images]
        
        Random(37).shuffle(self.img_label_pairs)

        self.tf = tf.Compose((tf.RandomHorizontalFlip(), tf.RandomRotation(10)))
        
    def __len__(self):
        return len(self.img_label_pairs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_label_pair = self.img_label_pairs[idx]
        data = io.read_image(img_label_pair[0], io.ImageReadMode.RGB)/255
        
        #Remove 12.81% top, 4 bottom, 4 left, 4 right
        crop_top = ceil(0.1281 * data.size(1))
        crop_bot = 4
        sub_vert = crop_top + crop_bot
        dims = (data.size(1)-sub_vert, data.size(2)-8)
        data = tff.crop(data, crop_top, 3, dims[0], dims[1])
        data = tff.resize(data, IMG_RES)
        data = data.float()

        return (data, torch.Tensor([img_label_pair[1]]).to(torch.float32))

def train():
    print('train_loss,train_acc,val_loss,val_acc,bad_acc,good_acc')

    model = tv.models.resnet34(num_classes=1)
    if USE_CUDA:
        model.cuda()
        
    dset = GoodBadWebcams(MAX_GOOD, MAX_BAD)
    train_set, val_set, test_set = random_split(dset, [0.70, 0.15, 0.15])
    train_loader = DataLoader(train_set, BATCH_SIZE, True, num_workers=4)
    val_loader = DataLoader(val_set, BATCH_SIZE, True, num_workers=4)
    test_loader = DataLoader(test_set, BATCH_SIZE, True, num_workers=4)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), LR)

    best_loss = 99999.0

    for epoch in range(EPOCHS):
        eprint('\nEpoch ' + str(epoch+1))
        eprint('Training...')
        
        model.train()
        train_set.dataset.augment = True
        
        running_loss = 0.0
        accuracy = BinaryAccuracy()
        
        count = 0
        
        for step, (data, labels) in enumerate(train_loader):
            bs =  labels.size(0)
            count += bs
            
            if USE_CUDA:
                data = data.cuda()
                labels = labels.cuda()
                
            output = torch.sigmoid(model(data))
            loss = loss_fn(output, labels)
            loss.backward()
            running_loss += labels.size(0) * loss.item()
            
            accuracy.update(torch.round(output).reshape((bs)), labels.reshape((bs)))
            
            optimizer.step()
            optimizer.zero_grad()
            
        print(running_loss/count, end=',')
        print(accuracy.compute().item(), end=',')
        
        
        eprint('\nValidating...')

        model.eval()
        val_loader.dataset.augment = False

        running_loss = 0.0
        accuracy = BinaryAccuracy()
        
        count = 0

        good = bad = goodAsBad = badAsGood = 0
        
        for step, (data, labels) in enumerate(val_loader):
            bs =  labels.size(0)
            count += bs
            
            if USE_CUDA:
                data = data.cuda()
                labels = labels.cuda()
                
            output = torch.sigmoid(model(data))
            loss = loss_fn(output, labels)
            running_loss += labels.size(0) * loss.item()
            
            accuracy.update(torch.round(output).reshape((bs)), labels.reshape((bs)))

            for i in range(bs):
                if round(labels[i].item()) == 0:
                    bad += 1
                    if round(output[i].item()) == 1:
                        badAsGood += 1
                else:
                    good += 1
                    if round(output[i].item()) == 0:
                        goodAsBad += 1

        val_loss = running_loss/count
        print(val_loss, end=',')
        print(accuracy.compute().item(), end=',')

        eprint('\nTesting...')

        model.eval()
        test_loader.dataset.augment = False

        running_loss = 0.0
        accuracy = BinaryAccuracy()
        
        count = 0

        good = bad = goodAsBad = badAsGood = 0
        
        for step, (data, labels) in enumerate(test_loader):
            bs =  labels.size(0)
            count += bs
            
            if USE_CUDA:
                data = data.cuda()
                labels = labels.cuda()
                
            output = torch.sigmoid(model(data))
            loss = loss_fn(output, labels)
            running_loss += labels.size(0) * loss.item()
            
            accuracy.update(torch.round(output).reshape((bs)), labels.reshape((bs)))

            for i in range(bs):
                if round(labels[i].item()) == 0:
                    bad += 1
                    if round(output[i].item()) == 1:
                        badAsGood += 1
                else:
                    good += 1
                    if round(output[i].item()) == 0:
                        goodAsBad += 1

        test_loss = running_loss/count
        print(test_loss, end=',')
        print(accuracy.compute().item(), end=',')

        print(1.0 - badAsGood/bad, end=',')
        print(1.0 - goodAsBad/good)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'goodbad-bestloss.pt')

def clean_dset():
    with torch.inference_mode():
        images = glob(path.normpath(DSET_DIR + '/**/*.png'), recursive=True)

        model = tv.models.resnet34(num_classes=1)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
        if USE_CUDA:
            model.cuda()

        # bar = Bar()
        # bar.max = len(images)

        good = open(path.join(LISTS_PATH, 'network-good.csv'), "a")
        bad = open(path.join(LISTS_PATH, 'network-bad.csv'), "a")

        for img in images:
            data = io.read_image(img, io.ImageReadMode.RGB)/255
            if USE_CUDA:
                data = data.cuda()

            #Remove 12.81% top, 4 bottom, 4 left, 4 right
            crop_top = ceil(0.1281 * data.size(1))
            crop_bot = 4
            sub_vert = crop_top + crop_bot
            dims = (data.size(1)-sub_vert, data.size(2)-8)
            data = tff.crop(data, crop_top, 3, dims[0], dims[1])
            data = tff.resize(data, IMG_RES)
            data = data.unsqueeze(0)
            data = data.float()

            output = torch.sigmoid(model(data))

            rel_path = str(Path(*Path(img).parts[-2:]))

            if round(output.item()) == 0:
                bad.write(rel_path+'\n')
            else:
                good.write(rel_path+'\n')

            # bar.next()

        good.close()
        bad.close()

def convert():
    INPUT_SHAPE = (1,3,280,280)
    OPTIMIZE = False
    IMG_PATH = '/home/feet/Documents/SchoolResearch/VisibilityResearch/Datasets/datasets/NewWebcams/4-9-24-6-47-PM/SITE7_ORNT312_VIS9mi.png'

    resize_func = util.image_cropping.get_resize_crop_fn(INPUT_SHAPE[-2:])

    class WithSigmoid(nn.Module):
        def __init__(self, model):
            super(WithSigmoid, self).__init__()
            self.model = model
        
        def forward(self, x):
            return nn.functional.sigmoid(self.model(x))
        
    
    model = tv.models.resnet34(num_classes=1)

    data = io.read_image(IMG_PATH, io.ImageReadMode.RGB)/255.0
    data = resize_func(data)
    data = torch.unsqueeze(data, 0)

    preconv = model(data)
    torch.onnx.export(model, data, "goodbad", input_names=['input'], output_names=['output'], opset_version=11)
    onnx_model = onnx.load("goodbad")
    tf_rep = prepare(onnx_model)
    tf_model_dir = "./tf_model"
    tf_rep.export_graph(tf_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)

    converter.optimizations = []
    if OPTIMIZE:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]


    tflite_model = converter.convert()

    with open("goodbad.tflite", "wb") as f:
        f.write(tflite_model)

    interpreter = tf.lite.Interpreter(model_path='goodbad.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    data = np.array(data, dtype=np.float32)

    input_data = np.array(data, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    postconv = interpreter.get_tensor(output_details[0]['index'])

    os.remove("goodbad")
    shutil.rmtree('tf_model')

    print(preconv)
    print(postconv)

train()