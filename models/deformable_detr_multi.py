# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc_multi import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer_multi import build_deforamble_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_sine_pos_embed(boxes, num_pos_feats=64):
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=boxes.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / num_pos_feats)

    x_embed = boxes[..., 0] * scale
    y_embed = boxes[..., 1] * scale
    w_embed = boxes[..., 2] * scale
    h_embed = boxes[..., 3] * scale
    
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_w = w_embed[..., None] / dim_t
    pos_h = h_embed[..., None] / dim_t

    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(-2)

    return torch.cat((pos_y, pos_x, pos_h, pos_w), dim=-1)

class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_obj_classes, num_pred_classes, num_rel_queries, num_queries, num_feature_levels, 
                 num_ref_frames = 3, aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_rel_queries = num_rel_queries
        self.num_ref_frames = num_ref_frames
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = MLP(hidden_dim, hidden_dim, 512, 2)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels

        self.temp_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.temp_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.rel_proj_head = MLP(hidden_dim, hidden_dim, 512, 2)
        self.sub_pointer_q = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.sub_pointer_k = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.obj_pointer_q = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.obj_pointer_k = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        self.rel_obj_attn = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1, batch_first=True)
        self.triplet_fusion = MLP(hidden_dim*3, hidden_dim, hidden_dim, 2)

        self.rel_query_embed = nn.Embedding(num_rel_queries, hidden_dim*2)
    
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
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
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.obj_logit_scale = nn.Parameter(torch.tensor([0.0]))
        self.rel_logit_scale = nn.Parameter(torch.tensor([0.0]))

        self.obj_class_bias = nn.Parameter(torch.ones(num_obj_classes) * bias_value)
        self.rel_class_bias = nn.Parameter(torch.ones(num_pred_classes) * bias_value)
        # ###############
        # self.temp_class_embed.bias.data = torch.ones(num_obj_classes + 1) * bias_value
        nn.init.constant_(self.temp_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.temp_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.temp_bbox_embed.layers[-1].bias.data[2:], -2.0)
        ##############

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor, text_embeddings: torch.Tensor, obj_text_embeddings: torch.Tensor, ref_frames: NestedTensor = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # print('features[-1].tensors.shape', features[-1].tensors.shape)

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
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = torch.cat([self.query_embed.weight, self.rel_query_embed.weight], dim=0)
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, final_hs, final_references_out = self.transformer(srcs, masks, pos, query_embeds, self.class_embed[-1])

        hs_obj = hs[:, :, :self.num_queries]
        hs_rel = hs[:, :, self.num_queries:]

        outputs_class = []
        obj_text_embeddings_norm = F.normalize(obj_text_embeddings.to(hs_obj.dtype), p=2, dim=-1)
        for c, x in zip(self.class_embed, hs_obj):
            obj_features_norm = F.normalize(c(x), p=2, dim=-1)
            obj_scale = self.obj_logit_scale.exp().clamp(max=20.0)
            logits = (torch.matmul(obj_features_norm, obj_text_embeddings_norm.T) * obj_scale) + self.obj_class_bias
            outputs_class.append(logits)
        outputs_class = torch.stack(outputs_class)

        outputs_coord = torch.stack([b(x).sigmoid() for b, x in zip(self.bbox_embed, hs_obj)])

        text_embeddings = text_embeddings.detach()
        obj_text_embeddings = obj_text_embeddings.detach()

        num_layers, bs, num_q, _ = hs_obj.shape

        hs_obj_flat = hs_obj.view(num_layers*bs, num_q, -1)
        hs_rel_flat = hs_rel.view(num_layers*bs, self.num_rel_queries, -1)
        coord_flat = outputs_coord.view(num_layers*bs, num_q, 4)

        geom_embed = get_sine_pos_embed(coord_flat.detach(), num_pos_feats=hs_obj.shape[-1]//4)
        obj_features_with_geom = hs_obj_flat + geom_embed

        hs_rel_flat = self.rel_obj_attn(hs_rel_flat, hs_obj_flat, hs_obj_flat)[0] + hs_rel_flat

        sub_q = self.sub_pointer_q(hs_rel_flat)
        sub_k = self.sub_pointer_k(obj_features_with_geom)
        outputs_sub_logits_flat = torch.bmm(sub_q, sub_k.transpose(-1,-2)) / math.sqrt(sub_q.size(-1))

        obj_q = self.obj_pointer_q(hs_rel_flat)
        obj_k = self.obj_pointer_k(obj_features_with_geom)
        outputs_obj_logits_flat = torch.bmm(obj_q, obj_k.transpose(-1,-2)) / math.sqrt(obj_q.size(-1))

        sub_feat = torch.bmm(outputs_sub_logits_flat.softmax(-1), hs_obj_flat)
        obj_feat = torch.bmm(outputs_obj_logits_flat.softmax(-1), hs_obj_flat)
        hs_rel_flat = self.triplet_fusion(torch.cat([hs_rel_flat, sub_feat, obj_feat], dim=-1))

        rel_features_flat = self.rel_proj_head(hs_rel_flat)
        rel_features_norm = F.normalize(rel_features_flat, p=2, dim=-1)
        text_embeddings_norm = F.normalize(text_embeddings.to(rel_features_norm.dtype), p=2, dim=-1)
        rel_scale = self.rel_logit_scale.exp().clamp(max=20.0)

        outputs_rel_class_flat = (torch.matmul(rel_features_norm, text_embeddings_norm.T) * rel_scale) + self.rel_class_bias

        outputs_sub_logits = outputs_sub_logits_flat.view(num_layers, bs, self.num_rel_queries, num_q)
        outputs_obj_logits = outputs_obj_logits_flat.view(num_layers, bs, self.num_rel_queries, num_q)
        outputs_rel_class = outputs_rel_class_flat.view(num_layers, bs, self.num_rel_queries, -1)
        
        # obj_features_with_geom = torch.cat([hs_obj, outputs_coord.detach()], dim=-1)

        # self.rel_obj_attn = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1, batch_first=True)
        # self.triplet_fusion = MLP(hidden_dim*3, hidden_dim, hidden_dim, 2)

        # hs_rel = self.rel_obj_attn(hs_rel, hs_obj, hs_obj)[0] + hs_rel

        # sub_q = self.sub_pointer_q(hs_rel)
        # sub_k = self.sub_pointer_k(obj_features_with_geom)
        # outputs_sub_logits = torch.matmul(sub_q, sub_k.transpose(-1, -2)) / math.sqrt(sub_q.size(-1))

        # obj_q = self.obj_pointer_q(hs_rel)
        # obj_k = self.obj_pointer_k(obj_features_with_geom)
        # outputs_obj_logits = torch.matmul(obj_q, obj_k.transpose(-1, -2)) / math.sqrt(obj_q.size(-1))

        # sub_feat = torch.bmm(outputs_sub_logits.softmax(-1), hs_obj)
        # obj_feat = torch.bmm(outputs_obj_logits.softmax(-1), hs_obj)
        # hs_rel = self.triplet_fusion(torch.cat([hs_rel, sub_feat, obj_feat], dim=-1))

        # rel_features = self.rel_proj_head(hs_rel)
        # rel_features_norm = F.normalize(rel_features, p=2, dim=-1)
        # text_embeddings_norm = F.normalize(text_embeddings.to(rel_features.dtype), p=2, dim=-1)
        # rel_scale = self.rel_logit_scale.exp().clamp(max=20.0)
        # outputs_rel_class = (torch.matmul(rel_features_norm, text_embeddings_norm.T) * rel_scale) + self.rel_class_bias

        # num_layers, bs, num_q = outputs_rel_class.shape[:3]

        # outputs_sub_logits = torch.matmul(self.sub_pointer(hs_rel), hs_obj.transpose(2,3))
        # outputs_obj_logits = torch.matmul(self.obj_pointer(hs_rel), hs_obj.transpose(2,3))

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1],
            'pred_rel_logits': outputs_rel_class[-1],
            'pred_sub_logits': outputs_sub_logits[-1],
            'pred_obj_logits': outputs_obj_logits[-1]
        }

        if self.aux_loss:
            out['aux_outputs'] = [
                {
                    'pred_logits': a,
                    'pred_boxes': b,
                    'pred_rel_logits': c,
                    'pred_sub_logits': d,
                    'pred_obj_logits': e
                }
                for a, b, c, d, e in zip(
                    outputs_class[:-1], outputs_coord[:-1],
                    outputs_rel_class[:-1], outputs_sub_logits[:-1], outputs_obj_logits[:-1]
                )
            ]

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:], outputs_coord[:])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_obj_classes, num_rel_classes, obj_matcher, rel_matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_obj_classes: number of object categories, omitting the special no-object category
            num_rel_classes: number of relation categories
            obj_matcher: module able to compute a matching between targets and proposals for objects
            rel_matcher: module able to compute a matching between targets and proposals for relations
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_obj_classes = num_obj_classes
        self.num_rel_classes = num_rel_classes
        self.obj_matcher = obj_matcher
        self.rel_matcher = rel_matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes_o = target_classes_o.clamp(0, src_logits.shape[-1] - 1)
        target_classes = torch.full(src_logits.shape, 0,
                                    dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes[idx[0], idx[1], target_classes_o] = 1

        # target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
        #                             dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o

        # target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
        #                                     dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        # target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # target_classes_onehot = target_classes_onehot[:,:,:-1]

        loss_ce = sigmoid_focal_loss(src_logits, target_classes, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_relation_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_rel_logits' in outputs
        src_logits = outputs['pred_rel_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["rel_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes_o = target_classes_o.clamp(0, src_logits.shape[-1] - 1)
        # bg_index = src_logits.shape[-1]
        target_classes = torch.full(src_logits.shape, 0, dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes[idx[0], idx[1], target_classes_o] = 1

        # target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
        #                                     dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        # target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        # target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_rel_ce = sigmoid_focal_loss(src_logits, target_classes, num_boxes, alpha=self.focal_alpha, gamma=2) 
        loss_rel_ce = loss_rel_ce * src_logits.shape[1]
        
        losses = {'loss_rel_ce': loss_rel_ce}
        return losses

        # num_logits_classes = src_logits.shape[-1]
        # empty_weight = torch.ones(num_logits_classes, device=src_logits.device)
        # empty_weight[-1] = 0.1
        # target_classes = target_classes.clamp(min=0, max=src_logits.shape[-1] - 1)

        # if empty_weight.shape[0] != src_logits.shape[-1]:
        #     empty_weight = torch.ones(src_logits.shape[-1], device=src_logits.device)
        #     empty_weight[-1] = 0.1
            
        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        # losses = {'loss_rel_ce': loss_ce}
        # return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.sigmoid().max(-1)[0] > 0.5).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if (target_boxes[:, 2:] <= 0).any():
            print("!!! Detected negative or zero width/height in target_boxes !!!")
            # import pdb; pdb.set_trace()

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_relation_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_sub_logits' in outputs and 'pred_obj_logits' in outputs
        idx = self._get_src_permutation_idx(indices)
        
        pred_boxes = outputs["pred_boxes"]
        src_sub_logits = outputs['pred_sub_logits'].softmax(-1)
        src_obj_logits = outputs['pred_obj_logits'].softmax(-1)

        src_sub_boxes = torch.matmul(src_sub_logits, pred_boxes)[idx]
        src_obj_boxes = torch.matmul(src_obj_logits, pred_boxes)[idx]

        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
        loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')

        losses = {}
        losses['loss_rel_bbox'] = (loss_sub_bbox.sum() + loss_obj_bbox.sum()) / num_boxes

        s_sub_xy = box_ops.box_cxcywh_to_xyxy(src_sub_boxes).clamp(0, 1)
        t_sub_xy = box_ops.box_cxcywh_to_xyxy(target_sub_boxes).clamp(0, 1)
        s_obj_xy = box_ops.box_cxcywh_to_xyxy(src_obj_boxes).clamp(0, 1)
        t_obj_xy = box_ops.box_cxcywh_to_xyxy(target_obj_boxes).clamp(0, 1)

        loss_sub_giou = 1 - torch.diag(box_ops.generalized_box_iou(s_sub_xy, t_sub_xy))
        loss_obj_giou = 1 - torch.diag(box_ops.generalized_box_iou(s_obj_xy, t_obj_xy))

        losses['loss_rel_giou'] = (loss_sub_giou.sum() + loss_obj_giou.sum()) / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
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

    def get_loss(self, loss, outputs, targets, indices_obj, indices_rel, num_boxes, num_rel_boxes, **kwargs):
        loss_map = {
            'labels': (self.loss_labels, indices_obj, num_boxes),
            'cardinality': (self.loss_cardinality, indices_obj, num_boxes),
            'boxes': (self.loss_boxes, indices_obj, num_boxes),
            'relation_labels': (self.loss_relation_labels, indices_rel, num_rel_boxes),
            'relation_boxes': (self.loss_relation_boxes, indices_rel, num_rel_boxes),
            'masks': (self.loss_masks, indices_obj, num_boxes)
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        loss_func, indices, normaliser = loss_map[loss]
        return loss_func(outputs, targets, indices, normaliser, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices_obj = self.obj_matcher(outputs_without_aux, targets)
        indices_rel = self.rel_matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_rel_boxes = sum(len(t["rel_labels"]) for t in targets)
        num_rel_boxes = torch.as_tensor([num_rel_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
            torch.distributed.all_reduce(num_rel_boxes)

        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        num_rel_boxes = torch.clamp(num_rel_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices_obj, indices_rel, num_boxes, num_rel_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_obj_aux = self.obj_matcher(aux_outputs, targets)
                indices_rel_aux = self.rel_matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss in ['labels', 'relation_labels']:
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_obj_aux, indices_rel_aux, num_boxes, num_rel_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices_obj_enc = self.obj_matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in ['masks', 'relation_labels', 'relation_boxes']:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices_obj_enc, None, num_boxes, num_rel_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        if 'pred_rel_logits' in outputs:
            pred_boxes = outputs["pred_boxes"]
            src_sub_logits = outputs['pred_sub_logits'].softmax(-1)
            src_obj_logits = outputs['pred_obj_logits'].softmax(-1)

            out_rel_logits = outputs['pred_rel_logits']
            out_sub_boxes = torch.matmul(src_sub_logits, pred_boxes)
            out_obj_boxes = torch.matmul(src_obj_logits, pred_boxes)

            rel_prob = rel_prob = out_rel_logits.softmax(-1)

            topk_rel_values, topk_rel_indexes = torch.topk(rel_prob.view(out_rel_logits.shape[0], -1), 100, dim=1)
            rel_scores = topk_rel_values

            topk_rel_boxes = topk_rel_indexes // rel_prob.shape[2]
            rel_labels = topk_rel_indexes % rel_prob.shape[2]

            sub_boxes = box_ops.box_cxcywh_to_xyxy(out_sub_boxes)
            sub_boxes = torch.gather(sub_boxes, 1, topk_rel_boxes.unsqueeze(-1).repeat(1,1,4))

            obj_boxes = box_ops.box_cxcywh_to_xyxy(out_obj_boxes)
            obj_boxes = torch.gather(obj_boxes, 1, topk_rel_boxes.unsqueeze(-1).repeat(1,1,4))

            sub_boxes = sub_boxes * scale_fct[:, None, :]
            obj_boxes = obj_boxes * scale_fct[:, None, :]

            for i in range(len(results)):
                results[i]['rel_scores'] = rel_scores[i]
                results[i]['rel_labels'] = rel_labels[i]
                results[i]['sub_boxes'] = sub_boxes[i]
                results[i]['obj_boxes'] = obj_boxes[i]

        return results

def _get_obj_labels_for_rel_queries(pred_logits_i, pred_sub_logits_i, pred_obj_logits_i, topk_rel_q):
    """Return predicted sub/obj class labels for each of the top-K relation queries.

    Args:
        pred_logits_i:     (num_obj_q, num_obj_classes)  – object class logits (sigmoid)
        pred_sub_logits_i: (num_rel_q, num_obj_q)        – pointer attention for subject
        pred_obj_logits_i: (num_rel_q, num_obj_q)        – pointer attention for object
        topk_rel_q:        (K,)                          – relation-query indices

    Returns:
        sub_labels: (K,) int64
        obj_labels: (K,) int64
    """
    # Hard assignment: which object query does each relation query point to?
    hard_sub = pred_sub_logits_i.argmax(dim=-1)  # (num_rel_q,)
    hard_obj = pred_obj_logits_i.argmax(dim=-1)  # (num_rel_q,)
    pred_classes = pred_logits_i.sigmoid().argmax(dim=-1)  # (num_obj_q,)
    sub_labels = pred_classes[hard_sub[topk_rel_q]]
    obj_labels = pred_classes[hard_obj[topk_rel_q]]
    return sub_labels, obj_labels


class PostProcessVidVRD(nn.Module):
    """SGDet postprocessor: model predicts boxes, object labels, and predicates."""
    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_boxes = outputs["pred_boxes"]
        src_sub_logits = outputs['pred_sub_logits'].softmax(-1)
        src_obj_logits = outputs['pred_obj_logits'].softmax(-1)

        out_rel_logits = outputs['pred_rel_logits']
        out_sub_boxes = torch.matmul(src_sub_logits, pred_boxes)
        out_obj_boxes = torch.matmul(src_obj_logits, pred_boxes)

        rel_prob = out_rel_logits.sigmoid()
        topk_rel_values, topk_rel_indices = torch.topk(rel_prob.view(out_rel_logits.shape[0], -1), 100, dim=1)

        scores = topk_rel_values
        topk_rel_boxes = topk_rel_indices // rel_prob.shape[2]   # relation-query index
        predicates = topk_rel_indices % rel_prob.shape[2]        # predicate-class index

        sub_boxes = box_ops.box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = torch.gather(sub_boxes, 1, topk_rel_boxes.unsqueeze(-1).repeat(1, 1, 4))

        obj_boxes = box_ops.box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = torch.gather(obj_boxes, 1, topk_rel_boxes.unsqueeze(-1).repeat(1, 1, 4))

        results = {}
        for i, target in enumerate(targets):
            vid_id = target.get('video_id', target.get('image_id', torch.tensor([0]))).item()
            frame_id = target.get('frame_id', torch.tensor([0])).item()

            sub_l, obj_l = _get_obj_labels_for_rel_queries(
                outputs['pred_logits'][i],
                outputs['pred_sub_logits'][i],
                outputs['pred_obj_logits'][i],
                topk_rel_boxes[i]
            )

            if vid_id not in results:
                results[vid_id] = {
                    'sub_trajectories': [{} for _ in range(100)],
                    'obj_trajectories': [{} for _ in range(100)],
                    'scores': scores[i],
                    'predicates': predicates[i],
                    'sub_labels': sub_l,
                    'obj_labels': obj_l,
                }

            scale_fct = (torch.stack([target['size'][1], target['size'][0],
                                      target['size'][1], target['size'][0]])
                         if 'size' in target
                         else torch.tensor([1, 1, 1, 1], device=scores.device))
            s_boxes = sub_boxes[i] * scale_fct
            o_boxes = obj_boxes[i] * scale_fct

            for k in range(100):
                results[vid_id]['sub_trajectories'][k][frame_id] = s_boxes[k].tolist()
                results[vid_id]['obj_trajectories'][k][frame_id] = o_boxes[k].tolist()

        return results


def _find_best_rel_query(pred_sub_logits_i, pred_obj_logits_i, pred_boxes_xyxy_i,
                         gt_sub_box, gt_obj_box):
    """Find the relation query that best attends to a given GT (subject, object) pair.

    Returns:
        best_k (int): index into relation queries
        combined_score (float): combined attention score
    """
    from util.box_ops import box_iou as _box_iou
    # IoU between object queries and GT boxes — both in the same normalised space
    iou_sub, _ = _box_iou(pred_boxes_xyxy_i, gt_sub_box.unsqueeze(0))  # (num_obj_q, 1)
    iou_obj, _ = _box_iou(pred_boxes_xyxy_i, gt_obj_box.unsqueeze(0))  # (num_obj_q, 1)
    best_sub_obj_q = iou_sub[:, 0].argmax()
    best_obj_obj_q = iou_obj[:, 0].argmax()

    sub_attn = pred_sub_logits_i.softmax(dim=-1)  # (num_rel_q, num_obj_q)
    obj_attn = pred_obj_logits_i.softmax(dim=-1)  # (num_rel_q, num_obj_q)

    combined = sub_attn[:, best_sub_obj_q] * obj_attn[:, best_obj_obj_q]  # (num_rel_q,)
    best_k = combined.argmax().item()
    return best_k, combined[best_k].item(), best_sub_obj_q.item(), best_obj_obj_q.item()


class PostProcessVidVRDSGCls(nn.Module):
    """SGCls postprocessor: GT trajectories given; model predicts object labels + predicates.

    For each GT relation pair visible in the current frame, we find the relation query
    that best attends to that pair and emit one prediction per (GT-pair, predicate-class).
    Predictions are keyed by (sub_tid, obj_tid, gt_pred_label) so that multiple GT
    relations between the same subject/object pair (different predicates) stay separate.
    """
    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_boxes = outputs["pred_boxes"]          # (bs, num_obj_q, 4) normalised cxcywh
        pred_logits = outputs['pred_logits']        # (bs, num_obj_q, num_obj_cls)
        pred_sub_logits = outputs['pred_sub_logits']  # (bs, num_rel_q, num_obj_q)
        pred_obj_logits = outputs['pred_obj_logits']  # (bs, num_rel_q, num_obj_q)
        out_rel_logits = outputs['pred_rel_logits']   # (bs, num_rel_q, num_pred_cls)

        results = {}
        for i, target in enumerate(targets):
            vid_id = target.get('video_id', target.get('image_id', torch.tensor([0]))).item()
            frame_id = target.get('frame_id', torch.tensor([0])).item()

            sub_boxes_gt = target.get('sub_boxes')   # (R, 4) normalised cxcywh — may be empty
            obj_boxes_gt = target.get('obj_boxes')
            sub_tids = target.get('sub_tids')
            obj_tids = target.get('obj_tids')
            rel_labels_gt = target.get('rel_labels')

            if sub_boxes_gt is None or sub_boxes_gt.shape[0] == 0:
                continue

            R = sub_boxes_gt.shape[0]
            pred_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[i])   # (num_obj_q, 4)
            pred_classes = pred_logits[i].sigmoid().argmax(dim=-1)          # (num_obj_q,)
            rel_prob = out_rel_logits[i].sigmoid()                          # (num_rel_q, P)

            scale_fct = (torch.stack([target['size'][1], target['size'][0],
                                      target['size'][1], target['size'][0]]).float()
                         if 'size' in target
                         else torch.ones(4, device=pred_boxes.device))

            for r in range(R):
                s_tid = sub_tids[r].item()
                o_tid = obj_tids[r].item()
                gt_pred = rel_labels_gt[r].item()
                key = (s_tid, o_tid, gt_pred)

                gt_sub_xyxy = box_ops.box_cxcywh_to_xyxy(sub_boxes_gt[r].unsqueeze(0))[0]
                gt_obj_xyxy = box_ops.box_cxcywh_to_xyxy(obj_boxes_gt[r].unsqueeze(0))[0]

                best_k, _, best_sub_q, best_obj_q = _find_best_rel_query(
                    pred_sub_logits[i], pred_obj_logits[i], pred_boxes_xyxy,
                    gt_sub_xyxy, gt_obj_xyxy
                )

                sub_label_model = pred_classes[best_sub_q].item()
                obj_label_model = pred_classes[best_obj_q].item()

                # GT boxes in pixel coords for trajectory
                gt_sub_pixel = (gt_sub_xyxy * scale_fct).tolist()
                gt_obj_pixel = (gt_obj_xyxy * scale_fct).tolist()

                if vid_id not in results:
                    results[vid_id] = {}

                if key not in results[vid_id]:
                    results[vid_id][key] = {
                        'sub_trajectory': {},
                        'obj_trajectory': {},
                        # Store per-predicate scores; take mean over frames for final score
                        'rel_prob_sum': rel_prob[best_k].clone(),
                        'frame_count': 1,
                        'sub_label': sub_label_model,
                        'obj_label': obj_label_model,
                    }
                else:
                    results[vid_id][key]['rel_prob_sum'] += rel_prob[best_k]
                    results[vid_id][key]['frame_count'] += 1

                results[vid_id][key]['sub_trajectory'][frame_id] = gt_sub_pixel
                results[vid_id][key]['obj_trajectory'][frame_id] = gt_obj_pixel

        return results


class PostProcessVidVRDPredCls(nn.Module):
    """PredCls postprocessor: GT trajectories + GT labels given; model predicts predicates only."""
    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_boxes = outputs["pred_boxes"]
        pred_sub_logits = outputs['pred_sub_logits']
        pred_obj_logits = outputs['pred_obj_logits']
        out_rel_logits = outputs['pred_rel_logits']

        results = {}
        for i, target in enumerate(targets):
            vid_id = target.get('video_id', target.get('image_id', torch.tensor([0]))).item()
            frame_id = target.get('frame_id', torch.tensor([0])).item()

            sub_boxes_gt = target.get('sub_boxes')
            obj_boxes_gt = target.get('obj_boxes')
            sub_tids = target.get('sub_tids')
            obj_tids = target.get('obj_tids')
            sub_labels_gt = target.get('sub_labels')
            obj_labels_gt = target.get('obj_labels')
            rel_labels_gt = target.get('rel_labels')

            if sub_boxes_gt is None or sub_boxes_gt.shape[0] == 0:
                continue

            R = sub_boxes_gt.shape[0]
            pred_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[i])
            rel_prob = out_rel_logits[i].sigmoid()   # (num_rel_q, P)

            scale_fct = (torch.stack([target['size'][1], target['size'][0],
                                      target['size'][1], target['size'][0]]).float()
                         if 'size' in target
                         else torch.ones(4, device=pred_boxes.device))

            for r in range(R):
                s_tid = sub_tids[r].item()
                o_tid = obj_tids[r].item()
                gt_pred = rel_labels_gt[r].item() if rel_labels_gt is not None else r
                key = (s_tid, o_tid, gt_pred)

                gt_sub_xyxy = box_ops.box_cxcywh_to_xyxy(sub_boxes_gt[r].unsqueeze(0))[0]
                gt_obj_xyxy = box_ops.box_cxcywh_to_xyxy(obj_boxes_gt[r].unsqueeze(0))[0]

                best_k, _, _, _ = _find_best_rel_query(
                    pred_sub_logits[i], pred_obj_logits[i], pred_boxes_xyxy,
                    gt_sub_xyxy, gt_obj_xyxy
                )

                gt_sub_pixel = (gt_sub_xyxy * scale_fct).tolist()
                gt_obj_pixel = (gt_obj_xyxy * scale_fct).tolist()

                if vid_id not in results:
                    results[vid_id] = {}

                if key not in results[vid_id]:
                    results[vid_id][key] = {
                        'sub_trajectory': {},
                        'obj_trajectory': {},
                        'rel_prob_sum': rel_prob[best_k].clone(),
                        'frame_count': 1,
                        'sub_label': sub_labels_gt[r].item(),
                        'obj_label': obj_labels_gt[r].item(),
                    }
                else:
                    results[vid_id][key]['rel_prob_sum'] += rel_prob[best_k]
                    results[vid_id][key]['frame_count'] += 1

                results[vid_id][key]['sub_trajectory'][frame_id] = gt_sub_pixel
                results[vid_id][key]['obj_trajectory'][frame_id] = gt_obj_pixel

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_obj_classes = args.num_obj_classes
    num_pred_classes = args.num_pred_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_obj_classes=num_obj_classes,
        num_pred_classes=num_pred_classes,
        num_rel_queries=args.num_rel_queries,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_ref_frames=args.num_ref_frames,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    obj_matcher, rel_matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    weight_dict['loss_rel_ce'] = args.rel_cls_loss_coef
    weight_dict['loss_rel_bbox'] = args.rel_bbox_loss_coef
    weight_dict['loss_rel_giou'] = args.rel_giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'relation_labels', 'relation_boxes']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_obj_classes, num_pred_classes, obj_matcher, rel_matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    if args.dataset_file == 'vidvrd':
        postprocessors['vidvrd'] = PostProcessVidVRD()
        postprocessors['vidvrd_sgcls'] = PostProcessVidVRDSGCls()
        postprocessors['vidvrd_predcls'] = PostProcessVidVRDPredCls()

    return model, criterion, postprocessors
