if __name__ == "__main__":
    # for debug only
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))
import torchvision.transforms.functional as F
from pathlib import Path
import random
from PIL import Image
import os
import os.path
from typing import Any, Callable, Optional, Tuple, List
import torch
import torch.utils.data
import torchvision
import cv2
from datasets.data_util import preparing_dataset
import numpy as np
import math
__all__ = ['build']

def rbox2vec(rboxes):
    angle = rboxes[:, 4:5]
    angle = angle /180 * np.pi
    c = rboxes[:, 0:2]
    w = rboxes[:, 2:3]
    h = rboxes[:, 3:4]
    p1 = c + np.concatenate([-np.cos(angle), -np.sin(angle)], axis=-1) * w / 2
    p2 = c + np.concatenate([np.sin(angle), -np.cos(angle)], axis=-1) * h / 2
    p3 = c + np.concatenate([np.cos(angle), np.sin(angle)], axis=-1) * w / 2
    p4 = c + np.concatenate([-np.sin(angle), np.cos(angle)], axis=-1) * h / 2
    ps = np.concatenate([p1, p2, p3, p4], axis=-1)
    return ps

def gaussian_label_cpu(label, num_class=360, u=0, sig=2.0):
    """
    转换成CSL Labels：
        用高斯窗口函数根据角度θ的周期性赋予gt labels同样的周期性，使得损失函数在计算边界处时可以做到“差值很大但loss很小”；
        并且使得其labels具有环形特征，能够反映各个θ之间的角度距离
    Args:
        label (float32):[1], theta class
        num_theta_class (int): [1], theta class num
        u (float32):[1], μ in gaussian function
        sig (float32):[1], σ in gaussian function, which is window radius for Circular Smooth Label
    Returns:
        csl_label (array): [num_theta_class], gaussian function smooth label
    """
    x = np.arange(-num_class/2, num_class/2)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))
    index = int(num_class/2 - label)
    return np.concatenate([y_sig[index:],
                           y_sig[:index]], axis=0)

def poly2csl(labels, num_cls_thata=360, radius=6.0, ignore_cls = [10, 12] , use_gaussian=False):
    """
    Trans poly format to rbox format.
    Args:
        polys (array): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
        num_cls_thata (int): [1], theta class num
        radius (float32): [1], window radius for Circular Smooth Label
        ignore_cls (list): True θ∈[-pi/2, pi/2) ， False

    Returns:
        use_gaussian True:
            rboxes (array):
            csl_labels (array): (num_gts, num_cls_thata)
        elif
            rboxes (array): (num_gts, [cx cy l s θ])
    """
    cls = labels[:, 0]
    polys = labels[:, 1:]
    assert polys.shape[-1] == 8
    if use_gaussian:
        csl_labels = []
    rboxes = []

    for poly, c in zip(polys, cls):
        poly = np.float32(poly.reshape(4, 2))
        (x, y), (w, h), angle = cv2.minAreaRect(poly) # θ ∈ [0， 90]
        if c in ignore_cls:
            angle = 0
        angle1 = gaussian_label_cpu(angle % 360, num_class=num_cls_thata, u=0, sig=radius)
        angle2 = gaussian_label_cpu((angle + 90) % 360, num_class=num_cls_thata, u=0, sig=radius)
        angle3 = gaussian_label_cpu((angle + 180) % 360, num_class=num_cls_thata, u=0, sig=radius)
        angle4 = gaussian_label_cpu((angle + 270) % 360, num_class=num_cls_thata, u=0, sig=radius)
        csl = angle1 + angle2 + angle3 + angle4
        csl_labels.append(csl)
        rboxes.append([x, y, w, h, angle])
    if use_gaussian:
        return np.array(rboxes), np.array(csl_labels)
    return np.array(rboxes)

def poly_filter(polys, h, w):
    """
    Filter the poly labels which is out of the image.
    Args:
        polys (array): (num, 8)

    Return：
        keep_masks (array): (num)
    """
    x = polys[:, 0::2]  # (num, 4)
    y = polys[:, 1::2]
    x_max = np.amax(x, axis=1)  # (num)
    x_min = np.amin(x, axis=1)
    y_max = np.amax(y, axis=1)
    y_min = np.amin(y, axis=1)

    x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0  # (num)
    keep_masks = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h)

    return keep_masks

def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(0, 0, 0))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(0, 0, 0))
    # Transform label coordinates
    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, 1:].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine
        targets_mask = poly_filter(polys=xy, h=height, w=width)
        targets[:, 1:] = xy
        targets = targets[targets_mask]
    return im, targets

class _CocoDetection(torchvision.datasets.vision.VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.indces = [i for i in range(len(self.ids))]
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        im = cv2.imread(os.path.join(self.root, path))
        return im

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class CocoDetection(_CocoDetection):
    def __init__(self, img_folder, ann_file, args, augment=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.augment = augment
        if self.augment:
            self.degrees = args.degrees
            self.translate = args.translate
            self.scale = args.scale
            self.shear = args.shear
            self.perspective = args.perspective
            self.hsv_h = args.hsv_h
            self.hsv_s = args.hsv_s
            self.hsv_v = args.hsv_v
            self.flipud = args.flipud
            self.aug_ratio = args.aug_ratio
            self.fliplr = args.fliplr
            self.mosaic = args.mosaic
            self.mosaic_border = args.mosaic_border
            self.mosaic_p = args.mosaic_p
            self.img_size = args.img_size
        else:
            self.mosaic = False
            self.img_size = args.img_size


    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indces, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            img, target = super(CocoDetection, self).__getitem__(index)
            h, w = img.shape[:2]
            if len(target) == 0:
                img_label = np.ones([0, 9])
            else:
                img_label = []
                for s_tgt in target:
                    img_label.append([s_tgt["category_id"]] + s_tgt["bbox"])
                img_label = np.array(img_label)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels = img_label.copy() # labels (array): (num_gt_perimg, [cls_id, poly])
            if labels.size:
                labels[:, [1, 3, 5, 7]] = img_label[:, [1, 3, 5, 7]] + padw
                labels[:, [2, 4, 6, 8]] = img_label[:, [2, 4, 6, 8]] + padh

            labels4.append(labels)
        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        h_filter = 2 * s
        w_filter = 2 * s
        labels_mask = poly_filter(polys=labels4[:, 1:].copy(), h=h_filter, w=w_filter)
        labels4 = labels4[labels_mask]
        # Augment
        img4, labels4 = random_perspective(img4, labels4,
                                           degrees=self.degrees,
                                           translate=self.translate,
                                           scale=self.scale,
                                           shear=self.shear,
                                           perspective=self.perspective,
                                           border=self.mosaic_border)  # border to remove

        return img4, labels4

    def Preprocess(self, img, target):
        img_h, img_w = img.shape[:2]
        img_new = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        length_max = max([img_h, img_w])
        img_w_new = int(self.img_size / length_max * img_w)
        img_h_new = int(self.img_size / length_max * img_h)
        img = cv2.resize(img, (img_w_new, img_h_new))
        img_new[0:img_h_new, 0:img_w_new, :] = img[:, :, :]
        ratio = self.img_size / length_max
        for i in range(len(target)):
            old_poly = target[i]["bbox"]
            if type(old_poly[0]) is list:
                old_poly = old_poly[0]
            new_poly = [coord * ratio for coord in old_poly]
            target[i]["bbox"] = new_poly
        return img_new, target
    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data.
        """
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        img_h, img_w = img.shape[:2]
        length_max = max([img_h, img_w])
        img, target = self.Preprocess(img, target)
        mosaic = self.mosaic and random.random() < self.mosaic_p
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(idx)
        else:
            if len(target) == 0:
                img_label = np.ones([0, 9])
            else:
                img_label = []
                for s_tgt in target:
                    img_label.append([s_tgt["category_id"]] + s_tgt["bbox"])
                img_label = np.array(img_label)
            labels = img_label.copy()

            if self.augment and random.random() < self.aug_ratio:
                img, labels = random_perspective(img, labels,
                                                      degrees=self.degrees,
                                                      translate=self.translate,
                                                      scale=self.scale,
                                                      shear=self.shear,
                                                      perspective=self.perspective)



        nl = len(labels)
        csl_labels = torch.zeros((nl, 360))
        if nl:
            rboxes, csl_labels = poly2csl(labels=labels[:, :], num_cls_thata=360, radius=2, use_gaussian=True)
            labels_mask = (rboxes[:, 0] >= 0) & (rboxes[:, 0] < img.shape[1]) \
                        & (rboxes[:, 1] >= 0) & (rboxes[:, 1] < img.shape[0]) \
                        & (rboxes[:, 2] > 5) & (rboxes[:, 3] > 5)

            new_lables = rbox2vec(rboxes)
            labels = np.concatenate([labels[:, 0:1], new_lables], axis=-1)
            csl_labels = csl_labels[labels_mask]
            labels = labels[labels_mask]
            nl = len(labels)
            csl_labels = torch.from_numpy(csl_labels)
        vecs = torch.from_numpy(labels[:, 1:])
        cls = torch.from_numpy(labels[:, 0])
        if nl:
            vecs = vecs.view(-1, 4, 2)
            s = torch.tensor([self.img_size, self.img_size]).unsqueeze(dim=0).unsqueeze(dim=0)
            vecs = vecs / s
            vecs = vecs.view(-1, 8)

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        out= {}
        out["vecs"] = vecs
        out["labels"] = cls.long()
        out["angles"] = csl_labels
        out["image_id"] = torch.tensor(image_id, dtype=torch.long)
        out["orig_size"] = torch.tensor([length_max, length_max], dtype=torch.long)
        return img, out

def build(image_set, args):
    root = Path(args.coco_path)
    mode = 'instances'
    PATHS = {
        "train": (root / "trainval2017", root / "annotations" / f'{mode}_trainval2017.json'),
        "train_reg": (root / "trainval2017", root / "annotations" / f'{mode}_trainval2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / f'{mode}_test2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    # copy to local path
    if os.environ.get('DATA_COPY_SHILONG') == 'INFO':
        preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)

    if image_set == "train":
        dataset = CocoDetection(img_folder, ann_file,
                                args=args,
                                augment=args.augment,
                                )
    else:
        dataset = CocoDetection(img_folder, ann_file,
                                args=args,
                                augment=False,
                                )
    return dataset



