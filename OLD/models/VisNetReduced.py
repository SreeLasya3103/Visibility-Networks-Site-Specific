import torch
import torch.nn as nn
import matplotlib
import math
import functools
import numpy as np

PC_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list('', ['#000000', '#3F003F', '#7E007E',
                                                                   '#4300BD', '#0300FD', '#003F82',
                                                                   '#007D05', '#7CBE00', '#FBFE00',
                                                                   '#FF7F00', '#FF0500'])
matplotlib.colors.LinearSegmentedColormap

class Model(nn.Module):
    def __init__(self, num_classes, num_channels, mean, std):
        super(Model, self).__init__()
 
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
        def conv_1(): 
            return [nn.Conv2d(num_channels, 32, 1),
                    nn.Conv2d(32, 32, 3),
                    nn.MaxPool2d(2, 2)]
        
        def conv_2(): 
            return [nn.Conv2d(32, 64, 1),
                    nn.Conv2d(64, 64, 3),
                    nn.MaxPool2d(2, 2)]
        
        def conv_3(): 
            return [nn.Conv2d(64, 128, 1),
                    nn.Conv2d(128, 128, 3, 2),
                    nn.Conv2d(128, 128, 1),
                    nn.MaxPool2d(2, 2)]
        
        linear_fft = [nn.Flatten(),
                      nn.LazyLinear(512),
                      nn.Dropout(0.4)]
        
        linear_pc_orig = [nn.Flatten(),
                          nn.LazyLinear(1024),
                          nn.Dropout(0.4)]
        
        linear = [nn.Linear(1536, 2048),
                  nn.Linear(2048, num_classes)]
        
        self.fft_1 = nn.Sequential(*conv_1())
        self.fft_2 = nn.Sequential(*conv_2())
        self.fft_3 = nn.Sequential(*conv_3())
        
        self.pc_1 = nn.Sequential(*conv_1())
        self.pc_2 = nn.Sequential(*conv_2())
        self.pc_3 = nn.Sequential(*conv_3())
        
        self.orig_1 = nn.Sequential(*conv_1())
        self.orig_2 = nn.Sequential(*conv_2())
        self.orig_3 = nn.Sequential(*conv_3())
        
        self.linear_fft = nn.Sequential(*linear_fft)
        self.linear_pc_orig = nn.Sequential(*linear_pc_orig)
        self.linear = nn.Sequential(*linear)
        
    def forward(self, x):
        x = (x - self.mean) / self.std
        x = x.permute((1, 0, 2, 3, 4))
        
        fft = self.fft_1(x[2])
        pc = self.pc_1(x[1])
        orig = self.orig_1(x[0])
        
        fft = torch.add(torch.add(pc, orig), fft)
        
        fft = self.fft_2(fft)
        pc = self.pc_2(pc)
        orig = self.orig_2(orig)
        
        fft = torch.add(torch.add(pc, orig), fft)
        
        fft = self.fft_3(fft)
        pc = self.pc_3(pc)
        orig = self.orig_3(orig)
        
        pc_orig = torch.add(pc, orig)
        
        fft = self.linear_fft(fft)
        pc_orig = self.linear_pc_orig(pc_orig)
        
        cat = torch.cat((fft, pc_orig), 1)
        
        return self.linear(cat)

@functools.cache
def highpass_mask(mask_radius, dim):
    mask = torch.ones(dim, dtype=torch.float32)
    mask_radius = np.multiply(dim, mask_radius)
    center = ((dim[0]-1)/2, (dim[1]-1)/2)
    center_tl = np.subtract(np.floor(center), mask_radius).astype(int)
    center_br = np.add(np.ceil(center), mask_radius).astype(int)
    
    for h in range(center_tl[0], center_br[0]):
        for w in range(center_tl[1], center_br[1]):
            h_dist = abs(h-center[0]) / mask_radius[0]
            w_dist = abs(w-center[1]) / mask_radius[1]
            distance = math.sqrt(h_dist**2 + w_dist**2)
            distance = min(1.0, distance)
            # if distance < 1.0:
            #     mask[h][w] = 0.0

            mask[h][w] = distance**6
    return mask

@functools.cache
def bandpass_mask(mask_radii, dim):
    mask = highpass_mask(mask_radii[0], dim) + lowpass_mask(mask_radii[1], dim)
    mask = (mask - 1.0) * -1.0

    return mask

@functools.cache
def lowpass_mask(mask_radius, dim):
    mask = (highpass_mask(mask_radius, dim) - 1.0) * -1.0

    return mask

def pass_filter(img, mask):
    orig_dim = (img.size(0), img.size(1))
    fft = torch.fft.fft2(img)
    fft = torch.fft.fftshift(fft)

    fft = fft*mask
    
    fft = torch.fft.ifftshift(fft)    
    fft = torch.fft.ifft2(fft, orig_dim)
    fft = fft.type(torch.float32)
    
    fft = torch.clamp(fft, 0.0, 1.0)
    
    return fft

def get_tf_function():
    def transform(img):
        img = img.repeat(3, 1, 1, 1)
        
        img[1] = torch.from_numpy(PC_CMAP(img[1][2])).permute((2,0,1))[:3,:,:]
        
        pass_points = (0.25, 0.125)

        # img[2][0] = pass_filter(img[2][2], lowpass_mask(pass_points[1], img[2][2].shape))
        # img[2][0] = (img[2][0] - torch.mean(img[2][0])) / torch.std(img[2][0])

        # img[2][1] = pass_filter(img[2][2], bandpass_mask(pass_points, img[2][2].shape))
        # img[2][1] = (img[2][1] - torch.mean(img[2][1])) / torch.std(img[2][1])
        
        img[2][2] = pass_filter(img[2][2], highpass_mask(pass_points[0], img[2][2].shape))
        img[2][2] = (img[2][2] - torch.mean(img[2][2])) / torch.std(img[2][2])

        # img[2][0] = torch.zeros(img[2][2].shape)
        # img[2][1] = torch.zeros(img[2][2].shape)
        # img[2][2] = torch.zeros(img[2][2].shape)

        # img[2][0] = lowpass_mask(pass_points[1], img[2][2].shape)
        # img[2][1] = bandpass_mask(pass_points, img[2][2].shape)
        # img[2][2] = highpass_mask(pass_points[0], img[2][2].shape)

        img[2] = torch.from_numpy(PC_CMAP(img[2][2])).permute((2,0,1))[:3,:,:]
        
        return img
    
    return transform