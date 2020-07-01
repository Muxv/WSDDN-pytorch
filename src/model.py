import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from torchvision.ops import roi_pool
from pretrained import VGG_CNN_F, VGG_CNN_M_1024, VGG_VD_1024

class WSDDN_S(nn.Module):
    def __init__(self):
        super(WSDDN_S, self).__init__()
        self.pretrain_net = VGG_CNN_F()
        self.pretrain_net.load_mat()

        self.roi_output_size = (6, 6)
        
        self.fc6 = nn.Linear(6*6*256, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8c = nn.Linear(4096, 20)
        self.fc8d = nn.Linear(4096, 20)
        
    def forward(self, x, regions, scores=None):
        #   x    : bs, c ,h, w
        # regions: bs, R, 4
        #  scores: bs, R
        regions = [regions[0]] # roi_pool require [Tensor(K, 4)]
        R = len(regions[0])
        out = self.pretrain_net(x) # bs, 256， h/16, w/16
        out = roi_pool(out, regions, self.roi_output_size, 1.0/16)  # R, 256, 6, 6
        out = out.view(R, -1)

        if scores:
            out = out * scores[0]

        out = F.relu(self.fc6(out))
        out = F.relu(self.fc7(out))
        # fc8x(out)   R, 20
        cls_score = F.softmax(self.fc8c(out), dim = 1)
        det_score = F.softmax(self.fc8d(out), dim = 0)
        combined = cls_score * det_score
        return combined
    