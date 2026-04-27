# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

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

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid() # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets]) 
            tgt_ids = tgt_ids.clamp(0, out_prob.shape[1] - 1)
            # print("tgt_ids_shape", tgt_ids.shape)
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            num_classes_predicted = out_prob.shape[-1] 
            if (tgt_ids >= num_classes_predicted).any():
                print(f"\n!!! CLASS LABEL ERROR !!!")
                print(f"Model predicts {num_classes_predicted} classes.")
                print(f"Found label IDs in dataset: {tgt_ids.unique().tolist()}")
                print(f"Offending IDs: {tgt_ids[tgt_ids >= num_classes_predicted].tolist()}")
                # import pdb; pdb.set_trace()

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            #print("pos_cost_class_shape", pos_cost_class.shape)
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            #print("cost_class_shape", cost_class.shape)

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            #print("size", sizes)
            indices = [linear_sum_assignment(C[i, :, sum(sizes[:k]):sum(sizes[:k])+s]) for k, (i, s) in enumerate(zip(range(bs), sizes))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class RelationHungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_rel_queries = outputs["pred_rel_logits"].shape[:2]

        out_prob = outputs["pred_rel_logits"].flatten(0, 1).sigmoid()
        pred_boxes = outputs["pred_boxes"]
        out_sub_logits = outputs["pred_sub_logits"].softmax(-1)
        out_obj_logits = outputs["pred_obj_logits"].softmax(-1)

        out_sub_bbox = torch.matmul(out_sub_logits, pred_boxes).flatten(0,1)
        out_obj_bbox = torch.matmul(out_obj_logits, pred_boxes).flatten(0,1)

        tgt_ids = torch.cat([v["rel_labels"] for v in targets])
        tgt_ids = tgt_ids.clamp(0, out_prob.shape[1] - 1)
        # if len(tgt_ids) == 0:
        #     print("WARNING: Batch has 0 relation targets")
        tgt_sub_bbox = torch.cat([v["sub_boxes"] for v in targets])
        tgt_obj_bbox = torch.cat([v["obj_boxes"] for v in targets])

        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_bbox, p=1)
        cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_bbox, p=1)
        
        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_bbox))
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_bbox))

        C = self.cost_bbox * (cost_sub_bbox + cost_obj_bbox) + \
            self.cost_class * cost_class + \
            self.cost_giou * (cost_sub_giou + cost_obj_giou)
        
        C = C.view(bs, num_rel_queries, -1).cpu()

        sizes = [len(v["sub_boxes"]) for v in targets]
        indices = [linear_sum_assignment(C[i, :, sum(sizes[:k]):sum(sizes[:k])+s]) for k, (i, s) in enumerate(zip(range(bs), sizes))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    obj_matcher = HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)
    rel_matcher = RelationHungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)
    
    return obj_matcher, rel_matcher