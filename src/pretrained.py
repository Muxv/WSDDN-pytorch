import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat

def copy(src_info, dst_layer, idx):
    """
    dst_layer : nn.Conv2d
    """
    layer = src_info[idx]
    kernel, bias = layer[0]['weights'][0][0]
    # in mat data  weight is w, h, c, bs
    dst_layer.weight.data.copy_(torch.from_numpy(kernel.transpose(3, 2, 1, 0)))
    dst_layer.bias.data.copy_(torch.from_numpy(bias.reshape(-1)))

class VGG_CNN_F(nn.Module):
    def __init__(self):
        super(VGG_CNN_F, self).__init__()
        self.conv1 = nn.Conv2d(3, 
                               64, 
                               kernel_size=11,
                               stride=4,
                               padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.lrn1 = nn.LocalResponseNorm(size=5,
                                         alpha=1e-4,
                                         beta=0.75,
                                         k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3,
                                  stride=2)

        self.conv2 = nn.Conv2d(64,
                               256,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.lrn2 = nn.LocalResponseNorm(size=5,
                                         alpha=1e-4,
                                         beta=0.75,
                                         k=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3,
                                  stride=2)
        self.conv3 = nn.Conv2d(256,
                               256,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256,
                               256,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256,
                               256,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.relu5 = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(self.lrn1(self.relu1(out)))
        
        out = self.conv2(out)
        out = self.pool2(self.lrn2(self.relu2(out)))

        out = F.relu(self.relu3(out))
        out = F.relu(self.relu4(out))
        out = F.relu(self.relu5(out))
        return out

    def load_mat(self, mat_path="../pretrained/imagenet-vgg-f.mat"):
        """
        idx    layer
        0       conv1
        4       conv2
        8       conv3
        10      conv4
        12      conv5
        """
        conv_idx = [0, 4, 8, 10, 12]
        conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        cnn_f_weights = loadmat(mat_path)
        layers_info = cnn_f_weights['layers'][0]
        for i in range(len(conv_idx)):
            copy(layers_info, conv_list[i], conv_idx[i])


class VGG_CNN_M_1024(nn.Module):
    def __init__(self):
        super(VGG_CNN_M_1024, self).__init__()
        self.conv1 = nn.Conv2d(3, 
                               96, 
                               kernel_size=7,
                               stride=2,
                               padding=0)
        self.lrn1 = nn.LocalResponseNorm(size=5,
                                         alpha=1e-4,
                                         beta=0.75,
                                         k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3,
                                  stride=2)

        self.conv2 = nn.Conv2d(96,
                               256,
                               kernel_size=5,
                               stride=2,
                               padding=1)
        self.lrn2 = nn.LocalResponseNorm(size=5,
                                         alpha=1e-4,
                                         beta=0.75,
                                         k=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3,
                                  stride=2)
        self.conv3 = nn.Conv2d(256,
                               512,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv4 = nn.Conv2d(512,
                               512,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv5 = nn.Conv2d(512,
                               512,
                               kernel_size=3,
                               stride=1,
                               padding=1)


    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(self.lrn1(F.relu(out)))
        
        out = self.conv2(out)
        out = self.pool2(self.lrn2(F.relu(out)))

        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        return out

    def load_mat(self, mat_path="../pretrained/imagenet-vgg-m-1024.mat"):
        """
        idx    layer
        0       conv1
        4       conv2
        8       conv3
        10      conv4
        12      conv5
        """
        conv_idx = [0, 4, 8, 10, 12]
        conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        cnn_f_weights = loadmat(mat_path)
        layers_info = cnn_f_weights['layers'][0]
        for i in range(len(conv_idx)):
            copy(layers_info, conv_list[i], conv_idx[i])

class VGG_VD_1024(nn.Module):
    def __init__(self):
        super(VGG_VD_1024, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        
        

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(F.relu(self.conv2(out)))
        
        out = F.relu(self.conv3(out))
        out = self.pool2(F.relu(self.conv4(out)))
        
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = self.pool3(F.relu(self.conv7(out)))
        
        out = F.relu(self.conv8(out))
        out = F.relu(self.conv9(out))
        out = self.pool4(F.relu(self.conv10(out)))
        
        out = F.relu(self.conv11(out))
        out = F.relu(self.conv12(out))
        out = F.relu(self.conv13(out))
        
        return out

    def load_mat(self, mat_path="../pretrained/imagenet-matconvnet-vgg-verydeep-16.mat"):
        conv_idx = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, 
                     self.conv5, self.conv6, self.conv7, self.conv8,
                     self.conv9, self.conv10, self.conv11, self.conv12,
                     self.conv13]
        cnn_f_weights = loadmat(mat_path)
        layers_info = cnn_f_weights['layers'][0]
        for i in range(len(conv_idx)):
            copy(layers_info, conv_list[i], conv_idx[i])

