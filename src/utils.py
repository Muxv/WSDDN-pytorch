import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 1e-5
EPOCHS = 20
SAVE_PATH = "../models/"
LOG_PATH = "../logs/"
OUT_PATH = "../outs/"
plt.rcParams['figure.dpi'] = 150 #分辨率

def draw_box(img, boxes):
    """
    img : PIL Image
    boxes: np.darray shape (N, 4)
    """
    p = np.asarray(img)
    for box in boxes:
        cv2.rectangle(p, (box[0], box[1]), (box[2], box[3]), (255, 255, 0))
    plt.imshow(p)