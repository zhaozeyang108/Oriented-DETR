# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py f
"""

import math
import os
import sys
from typing import Iterable
import cv2
import torch
import  numpy as np
import util.misc as utils
from util.utils import  to_device
from draw_color import color_table
from util.dota_eval import compute_metric
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break
    return


def poly2bbox(polys, angles):
    center = torch.mean(polys, dim=0)
    vecs = polys - center.unsqueeze(0)
    vecs = vecs.transpose(0, 1).unsqueeze(1)
    direction = torch.cat([torch.sin(angles * torch.pi / 180).unsqueeze(0), -torch.cos(angles * torch.pi / 180).unsqueeze(0)], dim = 0)
    direction = direction.unsqueeze(-1)
    matrix = torch.matmul(direction, vecs)
    matrix = torch.sum(matrix, dim=0)
    out = torch.max(matrix, dim=-1).values
    poly1 = center + out[0] * direction[:, 0, 0] + out[1] * direction[:, 1, 0]
    poly2 = center + out[1] * direction[:, 1, 0] + out[2] * direction[:, 2, 0]
    poly3 = center + out[2] * direction[:, 2, 0] + out[3] * direction[:, 3, 0]
    poly4 = center + out[3] * direction[:, 3, 0] + out[0] * direction[:, 0, 0]
    bbox = torch.cat([poly1, poly2, poly3, poly4])
    return bbox

def detect(model, postprocessors, data_loader, base_ds, device, args, logger=None):

    model.eval()

    class_names = base_ds.dataset["categories"]
    num_classes = len(class_names)
    output_per_class = {item + 1: [] for item in range(num_classes)}
    output_per_image = {}
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        _cnt += 1
        # if _cnt>300:
        #     break
        samples = samples.to(device)
        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, args.inf_conf)
        for i, result in enumerate(results):
            scores, labels, boxes, angles = result["scores"], result["labels"], result["boxes"], result["angles"]
            current_image_id = targets[i]["image_id"]
            scores = scores.cpu().numpy()
            if boxes.shape[0] != 0:
                recs = []
                points = []
                for box, angle in zip(boxes, angles):
                    rect = poly2bbox(box, angle)
                    rect = rect.cpu().numpy()
                    box = box.cpu().numpy()
                    recs.append(rect)
                    points.append(box)
                recs = np.array(recs)

            else:
                recs = np.ones([boxes.shape[0], 8])
                points=[]

            labels = labels.cpu().numpy()
            current_image_id = current_image_id.cpu().item()
            output_per_image[current_image_id] = [labels, scores, recs, points]
            for i in range(num_classes):
                class_id = i + 1
                boxes_current_class = recs[labels == class_id, :]
                scores_cureent_class = scores[labels == class_id]
                for i in range(len(boxes_current_class)):
                    output_per_class[class_id].append(
                        [current_image_id, scores_cureent_class[i], boxes_current_class[i]])

    cout = 0
    visdir = os.path.join(args.output_dir, "vis_box")
    txtdir = os.path.join(args.output_dir, "result_raw")
    if not os.path.exists(visdir):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(visdir)
    if not os.path.exists(txtdir):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(txtdir)

    for img_id, predict in output_per_image.items():
        img_info = base_ds.loadImgs(img_id)[0]
        # 获取图像的文件名和路径
        img_filename = img_info['file_name']
        img_path = os.path.join(args.coco_path, "test2017", img_filename)  # 你的图像目录
        # 读取和显示图像
        img = cv2.imread(img_path)
        categories = predict[0]
        polys = predict[2]
        points = predict[3]
        for cls, poly, pts in zip(categories, polys, points):
            poly = poly.reshape(-1, 1, 2).astype(int)
            img = cv2.polylines(img, [poly], True, color_table[cls-1], thickness=2)
        img_name = os.path.join(visdir, img_filename)
        cv2.imwrite(img_name, img)
        cout += 1
        if cout > 200:
            break

    for class_id, predict_all in output_per_class.items():
        class_name = class_names[class_id-1]["name"]
        for predict in predict_all:
            img_id, score, bbox = predict
            img_info = base_ds.loadImgs(img_id)[0]
            img_filename = img_info['file_name'].split(".")[0]
            with open(os.path.join(txtdir, "Task1_{}.txt".format(class_name)), 'a') as f:
                f.write(img_filename)
                f.write(" ")
                f.write(str(score.item()))
                f.write(" ")
                for temp in bbox:
                    f.write(str(temp.item()))
                    f.write(" ")
                f.write("\n")

    return result


@torch.no_grad()
def evaluate(model, postprocessors, data_loader, base_ds, device,args=None, logger=None):
    model.eval()
    class_names = base_ds.dataset["categories"]
    img_ids = base_ds.getImgIds()
    gt = {img_id: [] for img_id in img_ids}
    for img_id in img_ids:
        annotations = base_ds.loadAnns(base_ds.getAnnIds(imgIds=img_id))
        gt[img_id] = annotations
    num_classes = len(class_names)
    output_per_class = {item + 1: [] for item in range(num_classes)}
    output_per_image = {}
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'
    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        _cnt = _cnt + 1
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        for i, result in enumerate(results):
            scores, labels, boxes, angles = result["scores"], result["labels"], result["boxes"], result["angles"]
            current_image_id = targets[i]["image_id"]
            scores = scores.cpu().numpy()
            if boxes.shape[0] != 0:
                recs = []
                for box, angle in zip(boxes, angles):
                    rect = poly2bbox(box, angle)
                    rect = rect.cpu().numpy()
                    recs.append(rect)
                recs = np.array(recs)
            else:
                recs = np.ones([boxes.shape[0], 8])
            labels = labels.cpu().numpy()
            current_image_id = current_image_id.cpu().item()
            output_per_image[current_image_id] = [labels, scores, recs]
            for i in range(num_classes):
                class_id = i + 1
                boxes_current_class = recs[labels == class_id, :]
                scores_cureent_class = scores[labels == class_id]
                for i in range(len(boxes_current_class)):
                    output_per_class[class_id].append(
                        [current_image_id, scores_cureent_class[i], boxes_current_class[i]])
    result = compute_metric(gt, output_per_class, class_names, img_ids)
    result_075 = compute_metric(gt, output_per_class, class_names, img_ids, ovthresh=0.75)
    average_value = sum(result.values()) / len(result)
    average_value_75 = sum(result_075.values()) / len(result)
    print("average map50: {}".format(average_value))
    print("average map75: {}".format(average_value_75))
    return result
