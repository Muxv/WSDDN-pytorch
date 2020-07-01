import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import datetime

from model import WSDDN_S
from tqdm import tqdm
from utils import *
from torch import optim
from torchvision.ops import roi_pool, nms
from pretrained import VGG_CNN_F, VGG_CNN_M_1024, VGG_VD_1024
from VOCdatasets import VOCDectectionDataset

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


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Train WSDDN")
    parse.add_argument(
        "--propose", type=str, default='ssw', help="Use which way to propose ioR"
    )
    parse.add_argument(
        "--year", type=str, default='2007', help="Use which year of VOC"
    )
    args = parse.parse_args()

    propose_way = "selective_search" if args.propose == 'ssw' else "edge_box"

    trainval = VOCDectectionDataset("~/data/", args.year, 'trainval', region_propose=propose_way)
    train_loader = data.DataLoader(trainval, 1, shuffle=True)
    wsddn_s = WSDDN_S().to(DEVICE)
    wsddn_s.train()
    optimizer = optim.SGD(wsddn_s.parameters(), lr=LR, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    bce_loss = nn.BCELoss(reduction="sum")
    N = len(train_loader)

    log_file = LOG_PATH + "ssw_s_" + datetime.datetime.now().strftime('%m-%d_%H:%M') + ".txt"
    write_log(log_file, f"propose_way: {propose_way}")
    write_log(log_file, f"model_name: wsddn_s")
        
    for epoch in tqdm(range(EPOCHS), "Total"):
        epoch_loss = 0
        for img, gt_box, gt_target, regions, scores in tqdm(train_loader, f"Epoch {epoch}"):
            optimizer.zero_grad()
            # img   : Tensor(1, 3, h, w)
            # gt_tar: Tensor(1, R_gt)
            # region: Tensor(1, R, 4)
            img = img.to(DEVICE)
            regions = regions.to(DEVICE)
            gt_target = gt_target.to(DEVICE)
            if propose_way != "edge_box":
                scores = None
            else:
                scores = scores.to(DEVICE)

            combined = wsddn_s(img, regions, scores=scores)
            image_level_cls_score = torch.sum(combined, dim=0) # y
            out = bce_loss(image_level_cls_score, gt_target[0])

            epoch_loss += out.item()
            out.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss is {epoch_loss/N}")
        write_log(log_file, f"Epoch {epoch} Loss is {epoch_loss/N}")
        scheduler.step()
    torch.save(wsddn_s.state_dict(),
               SAVE_PATH + get_model_name(propose_way, args.year, "wsddn_s") + ".pt")
    write_log(log_file, f"model file is already saved")
    write_log(log_file, f"training finished")
        