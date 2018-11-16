# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """ Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class MRBrainNet(nn.Module):
    def __init__(self, n_classes=9):
        super(MRBrainNet, self).__init__()
        self.n_classes = n_classes
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # upsampling
        self.conv1_16 = nn.Conv2d(64, 16, 3, padding=1)
        
        self.conv2_16 = nn.Conv2d(128, 16, 3, padding=1)
        self.upscore_conv2 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        
        self.conv3_16 = nn.Conv2d(256, 16, 3, padding=1)
        self.upscore_conv3 = nn.ConvTranspose2d(16, 16, 4, 4)
        
        self.conv4_16 = nn.Conv2d(512, 16, 3, padding=1)
        self.upscore_conv4 = nn.ConvTranspose2d(16, 16, 8, 8)
        
        self.conv5_16 = nn.Conv2d(512, 16, 3, padding=1)
        self.upscore_conv5 = nn.ConvTranspose2d(16, 16, 16, 16)
        
        self.score = nn.Sequential(
                nn.Conv2d(4*16, self.n_classes, 1),
                nn.Dropout(0.5)
                )
        
        
    def forward(self, x):
        # Forward Propagation
        conv1 = x
        conv1 = self.relu1_1(self.conv1_1(conv1))
        conv1 = self.relu1_2(self.conv1_2(conv1))
        conv1 = self.pool1(conv1)
        
        conv2 = conv1
        conv2 = self.relu2_1(self.conv2_1(conv2))
        conv2 = self.relu2_2(self.conv2_2(conv2))
        conv2 = self.pool2(conv2)
        
        conv3 = conv2
        conv3 = self.relu3_1(self.conv3_1(conv3))
        conv3 = self.relu3_2(self.conv3_2(conv3))
        conv3 = self.relu3_3(self.conv3_3(conv3))
        conv3 = self.pool3(conv3)
        
        conv4 = conv3
        conv4 = self.relu4_1(self.conv4_1(conv4))
        conv4 = self.relu4_2(self.conv4_2(conv4))
        conv4 = self.relu4_3(self.conv4_3(conv4))
        conv4 = self.pool4(conv4)
        
        conv5 = conv4
        conv5 = self.relu5_1(self.conv5_1(conv5))
        conv5 = self.relu5_2(self.conv5_2(conv5))
        conv5 = self.relu5_3(self.conv5_3(conv5))
        conv5 = self.pool5(conv5)
        
        conv1_16 = self.conv1_16(conv1)
        conv2_16 = self.upscore_conv2(self.conv2_16(conv2))
        conv3_16 = self.upscore_conv3(self.conv3_16(conv3))
        conv4_16 = self.upscore_conv4(self.conv4_16(conv4))
        conv5_16 = self.upscore_conv5(self.conv5_16(conv5))
        
        final_layer = torch.cat([conv1_16, conv2_16, conv3_16, conv4_16], 1)
        score = self.score(final_layer)
        return score
        
    def _initialize_weights(self):
        pass
        # Initialize weights
        
    def copy_params_from_vgg16(self, vgg16):
        # Initialize network from pretrained vgg16
        features = [
                self.conv1_1, self.relu1_1,
                self.conv1_2, self.relu1_2,
                self.pool1,
                self.conv2_1, self.relu2_1,
                self.conv2_2, self.relu2_2,
                self.pool2,
                self.conv3_1, self.relu3_1,
                self.conv3_2, self.relu3_2,
                self.conv3_3, self.relu3_3,
                self.pool3,
                self.conv4_1, self.relu4_1,
                self.conv4_2, self.relu4_2,
                self.conv4_3, self.relu4_3,
                self.pool4,
                self.conv5_1, self.relu5_1,
                self.conv5_2, self.relu5_2,
                self.conv5_3, self.relu5_3,
                self.pool5
                ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
                
if __name__ == "__main__":
    x=torch.Tensor(4,3,256,256)
    model=MRBrainNet(n_classes=9)
    y=model(x)
    print(y.shape)
    #model = MRBrainNet(n_classes=9)
    #summary(model.to(torch.device("cpu")), (3, 240, 240))
    #summary(model.cuda(), (3, 256, 256))