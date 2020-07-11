import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np

from model import WSDDN_S, WSDDN_M
from tqdm import tqdm
from utils import *
from torchvision.ops import nms
from VOCdatasets import VOCDectectionDataset
from chainercv.evaluations import eval_detection_voc

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
    parse = argparse.ArgumentParser(description="validate WSDDN")
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

    testdata = VOCDectectionDataset("~/data/", args.year, 'test', region_propose=propose_way)
    test_loader = data.DataLoader(testdata, 1, shuffle=True)
    out = []
    wsddn = None
    det_threshold = 0.1
    if pretrained == 's':
        wsddn = WSDDN_S().to(DEVICE)
        
    elif pretrained == 'm':
        wsddn = WSDDN_M().to(DEVICE)
        
    wsddn.load_state_dict(torch.load(SAVE_PATH + get_model_name(propose_way, args.year, f"wsddn_{pretrained}") + ".pt"))
    
    log_file = LOG_PATH + f"Validate_{args.propose}_{pretrained}" + ".txt"
    write_log(log_file, f"propose_way: {propose_way}")
    write_log(log_file, f"model_name: wsddn_{pretrained}")
    
    total_pred_boxes = []
    total_pred_labels = []
    total_pred_scores = []
    total_true_boxes = []
    total_true_labels = []

    with torch.no_grad():
        wsddn.eval()
        for ten_imgs, gt, ten_regions, region, scores in tqdm(test_loader, "Evaluation"):
            region = region.to(DEVICE)
            if propose_way != "edge_box":
                scores = None
            else:
                scores = scores.to(DEVICE)
            avg_scores = torch.zeros((len(region[0]), 20), dtype=torch.float32)
            for i in range(4):
                per_img = ten_imgs[i].to(DEVICE)
                per_region = ten_regions[i].to(DEVICE)
                combined_scores = wsddn(per_img, per_region, scores)
                avg_scores += combined_scores.cpu()
            avg_scores /= 4
            
            gt = gt.numpy()[0]
            gt_boxex = gt[:, :4]
            gt_labels = gt[:, -1]
            
            per_pred_boxes = []
            per_pred_scores = []
            per_pred_labels = []
            
            region = region[0].cpu()
            
            for i in range(20):
                cls_scores = avg_scores[:, i]
                cls_region = region
                # ripe out the too smaller value
                score_filter = cls_scores > det_threshold
                cls_scores = cls_scores[score_filter]
                cls_region = cls_region[score_filter]
                                    

                nms_filter = nms(cls_region, cls_scores, 0.4)
                per_pred_boxes.append(cls_region[nms_filter].numpy())
                per_pred_scores.append(cls_scores[nms_filter].numpy())
                per_pred_labels.append(np.full(len(nms_filter), i, dtype=np.int32))
                
            total_pred_boxes.append(np.concatenate(per_pred_boxes, axis=0))
            total_pred_scores.append(np.concatenate(per_pred_scores, axis=0))
            total_pred_labels.append(np.concatenate(per_pred_labels, axis=0))
            total_true_boxes.append(gt_boxex)
            total_true_labels.append(gt_labels)
            
        result = eval_detection_voc(
            total_pred_boxes,
            total_pred_labels,
            total_pred_scores,
            total_true_boxes,
            total_true_labels,
            iou_thresh=0.5,
            use_07_metric=True,
        )

        write_log(log_file, f"Avg AP: {result['ap']}")
        write_log(log_file, f"Avg mAP: {result['map']}")