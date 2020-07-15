import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import datetime

from sklearn.metrics import average_precision_score
from model import *
from tqdm import tqdm
from utils import *
from torch import optim
from torchvision.ops import roi_pool, nms
from pretrained import VGG_CNN_F, VGG_CNN_M_1024, VGG_VD_1024
from datasets import VOCDectectionDataset


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Train WSDDN")
    parse.add_argument(
        "--propose", type=str, default='ssw', help="Use which way to propose ioR"
    )
    parse.add_argument(
        "--year", type=str, default='2007', help="Use which year of VOC"
    )
    parse.add_argument(
        "--pretrained", type=str, help="which pretrained model to use"
    )

    
    args = parse.parse_args()
    propose_way = "selective_search" if args.propose == 'ssw' else "edge_box"
    pretrained = args.pretrained.lower()

    trainval = VOCDectectionDataset("~/data/", args.year, 'trainval', region_propose=propose_way)
    train_loader = data.DataLoader(trainval, 1, shuffle=True)
    wsddn = None
    if pretrained == 'alexnet':
        wsddn = WSDDN_Alexnet().to(DEVICE)
    elif pretrained == 'vgg16':
        wsddn = WSDDN_VGG16().to(DEVICE)
        
    wsddn.train()
#     optimizer = optim.SGD(wsddn.parameters(), lr=LR, momentum=0.9)
    optimizer = optim.Adam(wsddn.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    bce_loss = nn.BCELoss(reduction="sum")
    N = len(train_loader)

    log_file = LOG_PATH + f"{args.propose}_{pretrained}_" + datetime.datetime.now().strftime('%m-%d_%H:%M') + ".txt"
    write_log(log_file, f"propose_way: {propose_way}")
    write_log(log_file, f"model_name: wsddn_{pretrained}")
    for epoch in tqdm(range(EPOCHS), "Total"):
        epoch_loss = 0.0
        y_pred = []
        y_true = []

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
            combined, fc7 = wsddn(img, regions, scores=scores)

            image_level_cls_score = torch.sum(combined, dim=0) # y
            # sum of combined may create value > 1.0
            image_level_cls_score = torch.clamp(image_level_cls_score, min=0.0, max=1.0)
            reg = alpha * wsddn.spatial_regulariser(regions[0], fc7, combined, gt_target[0])
            loss = bce_loss(image_level_cls_score, gt_target[0])
            out = loss + reg
            
            epoch_loss += out.item()
            out.backward()
            optimizer.step()

            y_pred.append(image_level_cls_score.detach().cpu().numpy().tolist())
            y_true.append(gt_target[0].detach().cpu().numpy().tolist())
        cls_ap = []
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        for i in range(20):
            cls_ap.append(average_precision_score(y_true[:,i], y_pred[:,i]))

        print(f"Epoch {epoch} classify AP is {str(cls_ap)}")
        write_log(log_file, f"Epoch {epoch} classify AP is {str(cls_ap)}")

        print(f"Epoch {epoch} classify mAP is {str(sum(cls_ap)/20)}")
        write_log(log_file, f"Epoch {epoch} classify mAP is {str(sum(cls_ap)/20)}")
        
        print(f"Epoch {epoch} Loss is {epoch_loss/N}")
        write_log(log_file, f"Epoch {epoch} Loss is {epoch_loss/N}")
        print("-" * 30)
        write_log(log_file, "-" * 30)
        scheduler.step()
    torch.save(wsddn.state_dict(),
               SAVE_PATH + get_model_name(propose_way, args.year, f"wsddn_{pretrained}") + ".pt")
    write_log(log_file, f"model file is already saved")
    write_log(log_file, f"training finished")
        