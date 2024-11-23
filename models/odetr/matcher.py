# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
def bbox2vec(bbox):
    center = (bbox[:, 0:2] + bbox[:, 2:4] + bbox[:, 4:6] + bbox[:, 6:8]) / 4
    v1 = bbox[:, 0:2] - center
    v2 = bbox[:, 2:4] - center
    v3 = bbox[:, 4:6] - center
    v4 = bbox[:, 6:8] - center
    mod1 = torch.sqrt(v1[:, 0:1] ** 2 + v1[:, 1:2] ** 2)
    mod2 = torch.sqrt(v2[:, 0:1] ** 2 + v2[:, 1:2] ** 2)
    mod3 = torch.sqrt(v3[:, 0:1] ** 2 + v3[:, 1:2] ** 2)
    mod4 = torch.sqrt(v4[:, 0:1] ** 2 + v4[:, 1:2] ** 2)
    v1 = v1 / mod1
    v2 = v2 / mod2
    v3 = v3 / mod3
    v4 = v4 / mod4
    vec = torch.cat([v1.unsqueeze(dim=-2), v2.unsqueeze(dim=-2), v3.unsqueeze(dim=-2), v4.unsqueeze(dim=-2)], dim=-2)
    mod = torch.cat([mod1, mod2, mod3, mod4], dim=-1)
    # vec = bbox - torch.cat([center for i in range(4)], dim=-1)
    return vec, mod, center

class AngleHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_angle: float = 1,
                 focal_alpha: float = 0.25):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_angle = cost_angle
        self.focal_alpha = focal_alpha
        assert cost_class != 0 or cost_bbox != 0 or cost_class != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            k = outputs["pred_boxes"].shape[2]  # number of points to represent an object
            indices = []
            for i in range(bs):
                # prepare for out
                out_prob = outputs["pred_logits"][i].sigmoid()
                out_points = outputs["pred_boxes"][i]  # [batch_size * num_queries, 4]
                out_angle = outputs["pred_angles"][i]
                out_center = out_points[:, k - 1, :]
                out_bbox = out_points[:, 0:k - 1, :]
                # prepare for target
                tgt_ids = targets[i]["labels"]
                tgt_bbox = targets[i]["vecs"]
                tgt_angle = targets[i]["angles"]
                tgt_vec, t_mod, t_center = bbox2vec(tgt_bbox)
                tgt_num = tgt_vec.shape[0]

                if tgt_num == 0:
                    cost_bbox = torch.ones(num_queries, tgt_num, device=out_bbox.device)
                else:
                    # calculate distance metrix in parallel
                    tgt_center = [x.unsqueeze(dim=0).repeat(num_queries, 1) for x in t_center]
                    tgt_center = torch.cat(tgt_center, dim=0).unsqueeze(1).repeat(1, k - 1, 1)
                    out_compute = out_bbox.repeat(tgt_num, 1, 1)
                    out_vec_compute = out_compute - tgt_center
                    tgt_vec_compute = [x.unsqueeze(dim=0).repeat(num_queries, 1, 1) for x in tgt_vec]
                    tgt_vec_compute = torch.cat(tgt_vec_compute, dim=0)
                    out_vec_compute = out_vec_compute.transpose(1, 2).unsqueeze(dim=2)
                    tgt_vec_compute = tgt_vec_compute.transpose(1, 2).unsqueeze(dim=-1)
                    dis_matrix = torch.matmul(tgt_vec_compute, out_vec_compute)
                    # calculate bbox cost
                    dis_matrix = torch.sum(dis_matrix, dim=1)
                    dis_matrix = torch.max(dis_matrix, dim=-1).values
                    dis_matrix = dis_matrix.unsqueeze(dim=1).chunk(tgt_num)
                    dis_matrix = torch.cat(dis_matrix, dim=1)
                    tgt_mod = t_mod.unsqueeze(dim=0).repeat(num_queries, 1, 1)
                    dis_matrix = torch.sqrt((dis_matrix - tgt_mod) ** 2)
                    cost_bbox = torch.sum(dis_matrix, dim=-1)
                # calculate center cost
                cost_center = torch.cdist(out_center.to(dtype=torch.float64), t_center, p=1)

                # calculate class cost
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

                # Compute the angle cost between angles
                cost_angle = torch.zeros(out_angle.shape[0], tgt_angle.shape[0]).to(cost_class.device)
                for j in range(tgt_angle.shape[0]):
                    tgt_tempt = tgt_angle[j]
                    cost_angle[:, j] = F.binary_cross_entropy_with_logits(out_angle,
                                                                          tgt_tempt.expand_as(out_angle),
                                                                          reduction='none').mean(1)
                # Final cost matrix
                C = self.cost_bbox * (cost_bbox + cost_center) + self.cost_class * cost_class + self.cost_angle * cost_angle
                indices.append(linear_sum_assignment(C.cpu()))
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    if args.matcher_type == 'AngleHungarianMatcher':
        return AngleHungarianMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_angle=args.set_cost_angle,
            focal_alpha=args.focal_alpha
        )
    else:
        raise NotImplementedError("Unknown args.matcher_type: {}".format(args.matcher_type))