import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from torchvision.ops import roi_pool
from pretrained import VGG_CNN_F, VGG_CNN_M_1024, VGG_VD_1024

def one2allbox_iou(target_box, others):
    """
     calculate the iou of box A to list of boxes
     target_box : Tensor()  1 * 4 
     others : Tensor()      N * 4 
     return  N * 1  ...iou
    
    """

    # get the min of xmax and ymax which organize the Intersection
    max_xy = torch.min(target_box[:, 2:], others[:, 2:]) 
    min_xy = torch.max(target_box[:, :2], others[:, :2])
    # get the xdistance and y distance
    # add 1 because distance = point2 - point1 + 1
    inter_wh = torch.clamp((max_xy - min_xy + 1), min=0)
    I = inter_wh[:, 0] * inter_wh[:, 1]
    A = (target_box[:, 2] - target_box[:, 0] + 1) * (target_box[:, 3] - target_box[:, 1] + 1)
    B = (others[:, 2] - others[:, 0] + 1) * (others[:, 3] - others[:, 1] + 1)
    return I / (A + B - I)

def spatial_regulariser(regions, fc7, combine_scores, labels):
    iou_th = 0.6
    K = 10 #  top 10 scores
    reg = 0
    positives = 0
    for c in range(20):
        # extract positive ones
        if labels[c].item() == 0:
            continue
        positives += 1
        topk_scores, topk_filter = combine_scores[:, c].topk(K, dim=0)
        topk_boxes = regions[topk_filter]
        topk_fc7 = fc7[topk_filter]
        
        # get box with the best box | iou > 0.6
        iou_mask = one2allbox_iou(topk_boxes[0:1, :], topk_boxes).view(K)
        iou_mask = (iou_mask > iou_th).float()
        
        fc7_diff = topk_fc7 - topk_fc7[0]
        score_diff = topk_scores.detach().view(K, 1)
        
        diff = fc7_diff * score_diff
        
        reg += 0.5 * (torch.pow(diff, 2).sum(1) * iou_mask).sum()
        
        reg /= positives
            
    return reg


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

        if scores is not None:
            out = out * scores[0] * 10

        out = F.relu(self.fc6(out))
        out = F.relu(self.fc7(out))
        
        fc7 = out 
        # fc8x(out)   R, 20
        cls_score = F.softmax(self.fc8c(out), dim = 1)
        det_score = F.softmax(self.fc8d(out), dim = 0)
        combined = cls_score * det_score

        return combined, fc7
    
class WSDDN_M(nn.Module):
    def __init__(self):
        super(WSDDN_M, self).__init__()
        self.pretrain_net = VGG_CNN_M_1024()
        self.pretrain_net.load_mat()

        self.roi_output_size = (6, 6)
        
        self.fc6 = nn.Linear(6*6*512, 4096)
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

        if scores is not None:
            out = out * scores[0] * 10
        fc7 = out
        out = F.relu(self.fc6(out))
        out = F.relu(self.fc7(out))
        # fc8x(out)   R, 20
        cls_score = F.softmax(self.fc8c(out), dim = 1)
        det_score = F.softmax(self.fc8d(out), dim = 0)
        combined = cls_score * det_score
        return combined, fc7
    
class WSDDN_L(nn.Module):
    def __init__(self):
        super(WSDDN_M, self).__init__()
        self.pretrain_net = VGG_VD_1024
        self.pretrain_net.load_mat()

        self.roi_output_size = (7, 7)
        
        self.fc6 = nn.Linear(6*6*512, 4096)
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

        if scores is not None:
            out = out * scores[0] * 10

        out = F.relu(self.fc6(out))
        out = F.relu(self.fc7(out))
        fc7 = out
        # fc8x(out)   R, 20
        cls_score = F.softmax(self.fc8c(out), dim = 1)
        det_score = F.softmax(self.fc8d(out), dim = 0)
        combined = cls_score * det_score
        return combined, fc7
    