# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP

from ..registry import MODULE_BUILD_FUNCS
class DINO(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False, iter_update=False,
                    query_dim=2, 
                    random_refpoints_xy=False,
                    fix_refpoints_hw=-1,
                    num_feature_levels=1,
                    nheads=8,
                    # two stage
                    dec_pred_class_embed_share=True,
                    dec_pred_bbox_embed_share=True,
                    dec_pred_angle_embed_share=True,
                    two_stage_class_embed_share=True,
                    two_stage_bbox_embed_share=True,
                    two_stage_angle_embed_share=True,
                    decoder_sa_type = 'sa',
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim)

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw


        # prepare input projection layers
        num_backbone_outs = len(backbone.num_channels)
        input_proj_list = []
        for _ in range(num_backbone_outs):
            in_channels = backbone.num_channels[_]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        for _ in range(num_feature_levels - num_backbone_outs):
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim),
            ))
            in_channels = hidden_dim
        self.input_proj = nn.ModuleList(input_proj_list)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = None


        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim//8 * 13, num_classes)
        _angle_embed = MLP(hidden_dim//8 * 13, hidden_dim//8 * 13, 360, 3)

        _class_squeeze = nn.Linear(hidden_dim, hidden_dim // 8)
        _angle_squeeze = nn.Linear(hidden_dim, hidden_dim // 8)

        _bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        _bbox_embed_sum = MLP(hidden_dim, hidden_dim, 2, 3)

        _class_embed_encoder = nn.Linear(hidden_dim, num_classes)
        _bbox_embed_encoder = MLP(hidden_dim, hidden_dim, 12, 3)
        _bbox_embed_encoder_sum = MLP(hidden_dim, hidden_dim, 2, 3)
        _angle_embed_encoder = MLP(hidden_dim, hidden_dim, 360, 3)

        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        _class_embed_encoder.bias.data = torch.ones(self.num_classes) * bias_value


        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(_bbox_embed_encoder.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed_encoder.layers[-1].bias.data, 0)
        nn.init.constant_(_bbox_embed_sum.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed_sum.layers[-1].bias.data, 0)
        nn.init.constant_(_bbox_embed_encoder_sum.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed_encoder_sum.layers[-1].bias.data, 0)



        nn.init.constant_(_angle_embed_encoder.layers[-1].weight.data, 0)
        nn.init.constant_(_angle_embed_encoder.layers[-1].bias.data, prior_prob)
        nn.init.constant_(_angle_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_angle_embed.layers[-1].bias.data, prior_prob)


        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
            box_embed_sum_layerlist = [_bbox_embed_sum for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
            box_embed_sum_layerlist = [copy.deepcopy(_bbox_embed_sum) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
            class_squeeze_layerlist = [_class_squeeze for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
            class_squeeze_layerlist = [copy.deepcopy(_class_squeeze) for i in range(transformer.num_decoder_layers)]
        if dec_pred_angle_embed_share:
            angle_embed_layerlist = [_angle_embed for i in range(transformer.num_decoder_layers)]
            angle_squeeze_layerlist = [_angle_squeeze for i in range(transformer.num_decoder_layers)]
        else:
            angle_embed_layerlist = [copy.deepcopy(_angle_embed) for i in range(transformer.num_decoder_layers)]
            angle_squeeze_layerlist = [copy.deepcopy(_angle_squeeze) for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.bbox_embed_sum = nn.ModuleList(box_embed_sum_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.angle_embed = nn.ModuleList(angle_embed_layerlist)
        self.class_squeeze = nn.ModuleList(class_squeeze_layerlist)
        self.angle_squeeze = nn.ModuleList(angle_squeeze_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.bbox_embed_sum = self.bbox_embed_sum
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.angle_embed = self.angle_embed

        if two_stage_bbox_embed_share:
            assert dec_pred_class_embed_share and dec_pred_bbox_embed_share and dec_pred_angle_embed_share
            self.transformer.enc_out_bbox_embed = _bbox_embed_encoder
            self.transformer.enc_out_bbox_embed_sum = _bbox_embed_encoder_sum
        else:
            self.transformer.enc_out_bbox_embed = _bbox_embed_encoder
            self.transformer.enc_out_bbox_embed_sum = _bbox_embed_encoder_sum

        if two_stage_class_embed_share:
            assert dec_pred_class_embed_share and dec_pred_bbox_embed_share and dec_pred_angle_embed_share
            self.transformer.enc_out_class_embed = _class_embed_encoder
        else:
            self.transformer.enc_out_class_embed = _class_embed_encoder
        if two_stage_angle_embed_share:
            assert dec_pred_class_embed_share and dec_pred_bbox_embed_share and dec_pred_angle_embed_share
            self.transformer.enc_out_angle_embed = _angle_embed_encoder
        else:
            self.transformer.enc_out_angle_embed = _angle_embed_encoder


        self.refpoint_embed = None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self, samples: NestedTensor, targets:List=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)


        assert targets is None


        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, masks, poss)
        # In case num object=0
        hs[0] += self.label_enc.weight[0, 0]*0.0

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_bbox_embed_sum, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, self.bbox_embed_sum, hs)):
            layer_delta_unsig_part = layer_bbox_embed(layer_hs[:, :, 0:12, :])
            layer_outputs_unsig_part = layer_delta_unsig_part + inverse_sigmoid(layer_ref_sig[:, :, 0:12, :])
            layer_delta_unsig_sum = layer_bbox_embed_sum(layer_hs[:, :, 12:13, :])
            layer_outputs_unsig_sum = layer_delta_unsig_sum + inverse_sigmoid(layer_ref_sig[:, :, 12:13, :])
            layer_outputs_unsig = torch.cat([layer_outputs_unsig_part, layer_outputs_unsig_sum], dim=-2)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        outputs_angle = torch.stack([layer_angle_embed(layer_angle_squeeze(layer_hs).flatten(-2)) for
                                     layer_angle_embed, layer_angle_squeeze, layer_hs in
                                     zip(self.angle_embed, self.angle_squeeze, hs)])
        outputs_class = torch.stack([layer_cls_embed(layer_cls_squeeze(layer_hs).flatten(-2)) for
                                     layer_cls_embed, layer_cls_squeeze, layer_hs in
                                     zip(self.class_embed, self.class_squeeze, hs)])

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1], "pred_angles": outputs_angle[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list, outputs_angle)
        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            interm_angle = self.transformer.enc_out_angle_embed(hs_enc[-1])
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord, 'pred_angles': interm_angle}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}
            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]


        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_angle):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, "pred_angles":c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_angle[:-1])]

class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_angles' in outputs
        src_logits = outputs['pred_angles']
        idx = self._get_src_permutation_idx(indices)
        target_angles_o = torch.cat([t["angles"][J] for t, (_, J) in zip(targets, indices)])
        querys_for_pre = src_logits[idx]
        angle_ce_loss = sigmoid_focal_loss(querys_for_pre, target_angles_o, num_boxes, alpha=0.8, gamma=2)
        losses = {'loss_angle': angle_ce_loss}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def bbox2vec(self, bbox):
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
        vec = torch.cat([v1.unsqueeze(dim=-2), v2.unsqueeze(dim=-2), v3.unsqueeze(dim=-2), v4.unsqueeze(dim=-2)],
                        dim=-2)
        mod = torch.cat([mod1, mod2, mod3, mod4], dim=-1)
        return vec, mod, center

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        src_center = src_boxes[:, 12, :]
        src_boxes = src_boxes[:, 0:12, :]
        target_boxes = torch.cat([t['vecs'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        tgt_vec, tgt_mod, tgt_cen = self.bbox2vec(target_boxes)
        tgt_center = tgt_cen.unsqueeze(1).repeat(1, 12, 1)
        src_vec = src_boxes - tgt_center
        src_vec_compute = src_vec.transpose(1, 2).unsqueeze(-2)
        tgt_vec_compute = tgt_vec.transpose(1, 2).unsqueeze(-1)
        src_mod = torch.matmul(tgt_vec_compute, src_vec_compute)
        src_mod = torch.sum(src_mod, dim=1)
        sample = torch.max(src_mod, dim=-1)
        src_mod = sample.values
        loss_bbox = F.l1_loss(src_mod, tgt_mod, reduction='none')
        loss_center = F.l1_loss(src_center.to(dtype=torch.float64), tgt_cen, reduction="none")
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes + loss_center.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            "angles": self.loss_angles
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self,dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
        assert pad_size % num_dn_groups==0
        single_pad=pad_size//num_dn_groups

        return output_known_lbs_bboxes,single_pad,num_dn_groups


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def rbox2poly(self,obboxes):
        """
        Trans rbox format to poly format.
        Args:
            rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

        Returns:
            polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
        """
        if isinstance(obboxes, torch.Tensor):
            center, w, h, theta = obboxes[:, :2], obboxes[:, 2:3], obboxes[:, 3:4], obboxes[:, 4:5]
            Cos, Sin = torch.cos(theta), torch.sin(theta)

            vector1 = torch.cat(
                (w / 2 * Cos, -w / 2 * Sin), dim=-1)
            vector2 = torch.cat(
                (-h / 2 * Sin, -h / 2 * Cos), dim=-1)
            point1 = center + vector1 + vector2
            point2 = center + vector1 - vector2
            point3 = center - vector1 - vector2
            point4 = center - vector1 + vector2
            order = obboxes.shape[:-1]
            return torch.cat(
                (point1, point2, point3, point4), dim=-1).reshape(*order, 8)
        else:
            center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
            Cos, Sin = np.cos(theta), np.sin(theta)

            vector1 = np.concatenate(
                [w / 2 * Cos, -w / 2 * Sin], axis=-1)
            vector2 = np.concatenate(
                [-h / 2 * Sin, -h / 2 * Cos], axis=-1)

            point1 = center + vector1 + vector2
            point2 = center + vector1 - vector2
            point3 = center - vector1 - vector2
            point4 = center - vector1 + vector2
            order = obboxes.shape[:-1]
            return np.concatenate(
                [point1, point2, point3, point4], axis=-1).reshape(*order, 8)
    @torch.no_grad()
    def forward(self, outputs, target_sizes, inf_conf=0.005):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox, out_angle = outputs['pred_logits'], outputs['pred_boxes'], outputs["pred_angles"]
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 400, dim=1)

        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        topk_angles = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        polys = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 13, 2))
        angles = torch.gather(out_angle, 1, topk_angles.unsqueeze(-1).repeat(1, 1, 360))
        _, theta_pred = torch.max(angles.sigmoid(), 2, keepdim=True)
        theta_pred1 = theta_pred
        theta_pred2 = (theta_pred + 90) % 360
        theta_pred3 = (theta_pred + 180) % 360
        theta_pred4 = (theta_pred + 270) % 360
        theta_pred = torch.cat([theta_pred1, theta_pred2, theta_pred3, theta_pred4], dim=-1)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_w, img_w, img_w, img_w, img_w, img_w, img_w, img_w, img_w, img_w, img_w, img_w], dim=1)
        polys = polys * scale_fct[:, None, :, None].cuda()
        results = []
        for s, l, b, a in zip(scores, labels, polys, theta_pred):
            valid_mask = s > inf_conf
            inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
            s = s[inds]
            l = l[inds]
            b = b[inds]
            a = a[inds]
            results.append({'scores': s, 'labels': l, 'boxes': b, "angles": a})

        return results

@MODULE_BUILD_FUNCS.registe_with_name(module_name='odetr')
def build_dino(args):
    num_classes = args.num_classes
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_deformable_transformer(args)
    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True
    try:
        dec_pred_angle_embed_share = args.dec_pred_angle_embed_share
    except:
        dec_pred_angle_embed_share = True

    model = DINO(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        dec_pred_angle_embed_share= dec_pred_angle_embed_share,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
    )
    matcher = build_matcher(args)
    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict["loss_angle"] = args.angle_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    interm_weight_dict = {}
    try:
        no_interm_box_loss = args.no_interm_box_loss
        no_interm_angle_loss = args.no_interm_angle_loss
    except:
        no_interm_box_loss = False
        no_interm_angle_loss =False
    _coeff_weight_dict = {
        'loss_ce': 1.0,
        'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
        "loss_angle":1.0 if not no_interm_angle_loss else 0.0
    }
    try:
        interm_loss_coef = args.interm_loss_coef
    except:
        interm_loss_coef = 1.0
    interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
    weight_dict.update(interm_weight_dict)
    losses = ['labels', 'boxes', "angles",  'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    return model, criterion, postprocessors
