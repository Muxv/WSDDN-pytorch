import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 1e-5
WD = 5e-4
alpha = 1e-3
EPOCHS = 20
SAVE_PATH = "../models/"
LOG_PATH = "../logs/"
OUT_PATH = "../outs/"
plt.rcParams['figure.dpi'] = 150 #分辨率
scTh = 1e-3

bce_weight_07 = torch.Tensor([0.05, 0.05, 0.04, 0.07, 0.05, 
                              0.07, 0.02, 0.04, 0.03, 0.09, 
                              0.06, 0.03, 0.04, 0.05, 0.01, 
                              0.05, 0.13, 0.05, 0.05, 0.05])

def draw_box(img, boxes):
    """
    img : PIL Image
    boxes: np.darray shape (N, 4)
    """
    p = np.asarray(img)
    for box in boxes:
        cv2.rectangle(p, (box[0], box[1]), (box[2], box[3]), (255, 255, 0))
    plt.imshow(p)
    
def get_model_name(propose_way, year, model_name):
    name = ""
    if propose_way == "selective_search":
        name += "ssw_"
    else:
        name += "eb_"
    
    name += str(year)+ "_" 
    name += model_name

    return name

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


def write_log(path, content):
    with open(path, 'a') as f:
        f.write(content + "\n")

def get_model_name(propose_way, year, model_name):
    name = ""
    if propose_way == "selective_search":
        name += "ssw_"
    else:
        name += "eb_"
    
    name += str(year)+ "_" 
    name += model_name
    return name