import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import copy

from scipy.io import loadmat


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def filter_small_boxes(boxes, min_size):
    """Filters out small boxes."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = (w >= min_size) & (h >= min_size)
    return mask

def totensor(img):
    t = transforms.ToTensor()
    return t(img)

def hflip_box(box, w):
    """
    box : [xmin, ymin, xmax, ymax]
    w   : image's width
    """
    box[0], box[2] = w - box[2], w - box[0]
    return 

def hflip_img(img):
    """
    img : PIL Image
    """
    fliper = transforms.RandomHorizontalFlip(1)
    return fliper(img)

def x2ychange_box(box):
    """
    box : [ymin, xmin, ymax, xmax]
     -> : [xmin, ymin, xmax, ymax]
    
    """
    box[0], box[1] = box[1], box[0]
    box[2], box[3] = box[3], box[2]
    return 

def resize_box(box, ratio):
    """
    box    : [xmin, ymin, xmax, ymax]
    ration : change ratio
    
    """    
    box[0] = int(ratio * box[0])
    box[1] = int(ratio * box[1])
    box[2] = int(ratio * box[2])
    box[3] = int(ratio * box[3])
    
def resize_img_maxlength(img, maxlength):
    """
    img : PIL Image
    maxlength : change the image max side length to 
    """
    w, h = img.size
    if (w > h):
        resizer = transforms.Resize((int(maxlength*h/w), maxlength))
        ratio = maxlength/w
    else: # h >= w
        resizer = transforms.Resize((maxlength, int(maxlength*w/h)))
        ratio = maxlength/h
    img = resizer(img)
    return img, ratio

class VOCAnnotationAnalyzer():
    """
    deal with annotation data (dict)
    
    Arguments:
        cls_to_idx (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """
    def __init__(self, cls_to_idx=None, keep_difficult=False):
        self.cls_to_idx = cls_to_idx or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult
        
    def __call__(self, annotation: dict):
        w = int(annotation['size']['width'])
        h = int(annotation['size']['height'])
        # if img only contains one gt that annotation['object'] is just a dict, not a list
        objects = [annotation['object']] if type(annotation['object']) != list else annotation['object']
        res = [] # [xmin, ymin, xmax, ymax, label]
        for box in objects:
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            difficult = int(box['difficult'])
            if not self.keep_difficult and difficult:
                continue
            name = box['name']
            bnd = []
            for pt in pts:
                bnd.append(int(box['bndbox'][pt]))
            bnd.append(self.cls_to_idx[name])
            res.append(bnd)
            
        return res
            
        
class VOCDectectionDataset(data.Dataset):
    def __init__(self, root, year, image_set,
                 transform=None, 
                 target_transform=VOCAnnotationAnalyzer(),
                 dataset_name='VOC07_12',
                 region_propose='selective_search'):
        super(VOCDectectionDataset, self).__init__()
        self.datas = datasets.VOCDetection(root, str(year), image_set, download=False)
        self.image_set = image_set
        self.transform = transform
        self.name = dataset_name
        self.target_transform = target_transform # use for annotation
        self.longer_sides = [480, 576, 688, 864, 1200]
        if region_propose not in ['selective_search', 'edge_box']:
            raise NotImplementedError(f'{region_propose} not Supported')

        self.region_propose = region_propose
        self.box_mat = self.get_mat(year, image_set, region_propose)
            
            
    def get_box_from_mat(self, index):
        return self.box_mat['boxes'][0][index].tolist()

    def get_boxScore_from_mat(self, index):
        score = None
        if self.region_propose == 'edge_box':
            score = self.box_mat['boxScores'][0][index].tolist()
        return score
    
    def get_mat(self, year, image_set, region_propose):
        """
        load the box generated
        """
        boxes = None
        boxes_score = 0
        
        if str(year) == '2007' and image_set == 'trainval' and region_propose == 'selective_search':
            mat = loadmat("../region/SelectiveSearchVOC2007trainval.mat")
        elif str(year) == '2007' and image_set == 'test' and region_propose == 'selective_search':
            mat = loadmat("../region/SelectiveSearchVOC2007test.mat")
        if str(year) == '2007' and image_set == 'trainval' and region_propose == 'edge_box':
            mat = loadmat("../region/EdgeBoxesVOC2007trainval.mat")
        elif str(year) == '2007' and image_set == 'test' and region_propose == 'edge_box':
            mat = loadmat("../region/EdgeBoxesVOC2007test.mat")
        return mat
            
class VOCDectectionDataset(data.Dataset):
    def __init__(self, root, year, image_set,
                 transform=None, 
                 target_transform=VOCAnnotationAnalyzer(),
                 dataset_name='VOC07_12',
                 region_propose='selective_search'):
        super(VOCDectectionDataset, self).__init__()
        self.datas = datasets.VOCDetection(root, str(year), image_set, download=False)
        self.image_set = image_set
        self.transform = transform
        self.name = dataset_name
        self.target_transform = target_transform # use for annotation
        self.longer_sides = [480, 576, 688, 864, 1200]
        if region_propose not in ['selective_search', 'edge_box']:
            raise NotImplementedError(f'{region_propose} not Supported')

        self.region_propose = region_propose
        self.box_mat = self.get_mat(year, image_set, region_propose)
            
            
    def get_box_from_mat(self, index):
        return self.box_mat['boxes'][0][index].tolist()

    def get_boxScore_from_mat(self, index):
        score = None
        if self.region_propose == 'edge_box':
            score = self.box_mat['boxScores'][0][index].tolist()
        return score
    
    def get_mat(self, year, image_set, region_propose):
        """
        load the box generated
        """
        boxes = None
        boxes_score = None
        
        if str(year) == '2007' and image_set == 'trainval' and region_propose == 'selective_search':
            mat = loadmat("../region/SelectiveSearchVOC2007trainval.mat")
        elif str(year) == '2007' and image_set == 'test' and region_propose == 'selective_search':
            mat = loadmat("../region/SelectiveSearchVOC2007test.mat")
        if str(year) == '2007' and image_set == 'trainval' and region_propose == 'edge_box':
            mat = loadmat("../region/EdgeBoxesVOC2007trainval.mat")
        elif str(year) == '2007' and image_set == 'test' and region_propose == 'edge_box':
            mat = loadmat("../region/EdgeBoxesVOC2007test.mat")
        return mat
            
    def __getitem__(self, index):
        img, gt = self.datas[index]
        region = self.get_box_from_mat(index)
        region_score = self.get_boxScore_from_mat(index)
        if self.target_transform:
            gt = self.target_transform(gt["annotation"])
        w, h = img.size
        # ----------------------------------------------------------------------------------
        if self.image_set == "trainval":
            if self.transform is None:
                # follow by paper: randomly horiztontal flip and randomly resize
                for box in region:
                    x2ychange_box(box)                
                if np.random.random() > 0.5: # then flip
                    img = hflip_img(img)
                    for box in gt: # change gt
                        hflip_box(box, w)
                    for box in region: # ssw generate is [ymin, xmin, ymax, xmax]
                        hflip_box(box, w)

                # then resize
                max_side = self.longer_sides[np.random.randint(5)]
                img, ratio = resize_img_maxlength(img, max_side)
                for box in gt:
                    resize_box(box, ratio)
                for box in region:
                    resize_box(box, ratio)
            else:
                raise NotImplementedError("This dataset can only be compatible with the paper's implementation")

            img = totensor(img)
            gt = np.array(gt)
            gt_box = np.array(gt[:, :4])
            gt_target = gt[:, -1]
            target = [0 for _ in range(len(VOC_CLASSES))]
            for t in gt_target:
                target[t] = 1.0
            
            gt_target = np.array(target).astype(np.float32)
            gt_box = np.array(gt).astype(np.float32) # split gt -> gt_box,  gt_target
            
            region = np.array(region).astype(np.float32)

            region_filter = filter_small_boxes(region, 32)
            region = region[region_filter]
            
            if region_score:
                region_score = np.array(region_score)
                
            if region_score is None:
                return img, gt_box, gt_target, region, np.array([0])
            else:
                return img, gt_box, gt_target, region, region_score
        # ----------------------------------------------------------------------------------

        elif self.image_set == "test":
            ten_images = []
            ten_gt = []
            ten_regions = []
            if self.transform is not None:
                raise NotImplementedError("This dataset can only be compatible with the paper's implementation")
            else:
                # first change box's cor
                for box in region:
                    x2ychange_box(box)
                # follow by paper: get 10 images for hflip and resize2 5sizes
                for flip in [0.0, 1.0]:
                    fliper = transforms.RandomHorizontalFlip(flip)
                    for max_side in self.longer_sides:
                        new_img = img.copy()
                        new_gt = copy.deepcopy(gt)
                        new_region = copy.deepcopy(region)
                        if flip == 1.0:
                            new_img = hflip_img(new_img)
                            for box in new_gt: # change gt
                                hflip_box(box, w)
                            for box in new_region: # ssw generate is [ymin, xmin, ymax, xmax]
                                hflip_box(box, w)
                        new_img, ratio = resize_img_maxlength(new_img, max_side)
                        for box in new_gt:
                            resize_box(box, ratio)
                        for box in new_region:
                            resize_box(box, ratio)

                        ten_images.append(totensor(new_img))
                        ten_gt.append(np.array(new_gt).astype(np.float32))
                        # ripe out too small boxes
                        new_region = np.array(new_region).astype(np.float32)
                        region_filter = filter_small_boxes(new_region, 32)
                        new_region = new_region[region_filter]
                        ten_regions.append(new_region)
            if region_score:
                region_score = np.array(region_score)
            if region_score is None:
                return ten_images, ten_gt, ten_regions, np.array([0])
            else:
                return ten_images, ten_gt, ten_regions, region_score
            
        else:
            raise ValueError(f"image_set can only be 'test' or 'trainval'")
        
    def __len__(self):
        return len(self.datas)