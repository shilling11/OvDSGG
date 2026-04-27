# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    # backbone, 是否训练backbone, 是否返回中间值
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        # tensor list 
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class CLIPViTBackbone(nn.Module):
    """CLIP ViT visual encoder producing spatial patch features aligned to CLIP text space."""

    CONFIGS = {
        'clip_vitb16': ('ViT-B/16', 768, 16),
        'clip_vitl14': ('ViT-L/14', 1024, 14),
    }

    def __init__(self, name: str, train_backbone: bool, n_visual_ctx: int = 0):
        super().__init__()
        import clip
        clip_name, embed_dim, patch_size = self.CONFIGS[name]
        model, _ = clip.load(clip_name, device='cpu')
        self.visual = model.visual
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.strides = [patch_size]
        self.num_channels = [embed_dim]

        # Freeze all by default; optionally fine-tune last 4 blocks + ln_post
        for p in self.visual.parameters():
            p.requires_grad_(False)
        if train_backbone:
            for block in self.visual.transformer.resblocks[-4:]:
                for p in block.parameters():
                    p.requires_grad_(True)
            for p in self.visual.ln_post.parameters():
                p.requires_grad_(True)

        # Visual prompt tokens (VPT-shallow): injected between CLS and patch tokens.
        # Stored directly on CLIPViTBackbone (NOT under self.visual) so the freeze
        # loop above does not affect them — they are always trainable.
        self.n_visual_ctx = n_visual_ctx
        if n_visual_ctx > 0:
            self.visual_prompt_tokens = nn.Parameter(
                torch.empty(n_visual_ctx, embed_dim).normal_(std=0.02)
            )
            self.visual_prompt_pos = nn.Parameter(torch.zeros(n_visual_ctx, embed_dim))

    def _interpolate_pos_embed(self, ph: int, pw: int, dtype, device):
        """Interpolate CLIP positional embeddings from training size to (ph, pw) patch grid."""
        pos = self.visual.positional_embedding  # [1 + H0*W0, D]
        cls_pos = pos[:1]                       # [1, D]
        patch_pos = pos[1:]                     # [H0*W0, D]
        orig_side = int(patch_pos.shape[0] ** 0.5)
        # reshape to spatial, interpolate, reshape back
        patch_pos = patch_pos.reshape(1, orig_side, orig_side, -1).permute(0, 3, 1, 2).float()
        patch_pos = F.interpolate(patch_pos, size=(ph, pw), mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(ph * pw, -1)
        return torch.cat([cls_pos, patch_pos], dim=0).to(dtype=dtype, device=device)  # [1+ph*pw, D]

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors          # [B, 3, H, W]
        B, _, H, W = x.shape
        ph = H // self.patch_size
        pw = W // self.patch_size
        v = self.visual

        # Patch embedding: conv1 maps [B,3,H,W] → [B,D,ph,pw]
        x = v.conv1(x)                                           # [B, D, ph, pw]
        x = x.reshape(B, self.embed_dim, -1).permute(0, 2, 1)   # [B, ph*pw, D]

        # Prepend CLS token
        cls = v.class_embedding.to(x.dtype).expand(B, 1, -1)    # [B, 1, D]
        x = torch.cat([cls, x], dim=1)                           # [B, 1+ph*pw, D]

        # Insert visual prompt tokens between CLS and patch tokens
        if self.n_visual_ctx > 0:
            prompts = self.visual_prompt_tokens.to(x.dtype).unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)  # [B, 1+n_ctx+ph*pw, D]

        # Add (interpolated) positional embedding
        pos_embed = self._interpolate_pos_embed(ph, pw, x.dtype, x.device)  # [1+ph*pw, D]
        if self.n_visual_ctx > 0:
            prompt_pos = self.visual_prompt_pos.to(x.dtype)       # [n_ctx, D]
            full_pos = torch.cat([pos_embed[:1], prompt_pos, pos_embed[1:]], dim=0)
            x = x + full_pos
        else:
            x = x + pos_embed

        # Transformer (CLIP uses LND convention)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        for block in v.transformer.resblocks:
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        x = x.permute(1, 0, 2)

        # Drop CLS and prompt tokens; apply ln_post to patch tokens only
        if self.n_visual_ctx > 0:
            patch_feats = v.ln_post(x[:, 1 + self.n_visual_ctx:, :])  # [B, ph*pw, D]
        else:
            patch_feats = v.ln_post(x[:, 1:, :])                      # [B, ph*pw, D]
        patch_feats = patch_feats.permute(0, 2, 1).reshape(B, self.embed_dim, ph, pw)

        # Rebuild spatial mask
        m = tensor_list.mask                                     # [B, H, W]
        mask = F.interpolate(m[None].float(), size=(ph, pw)).to(torch.bool)[0]

        return {"0": NestedTensor(patch_feats, mask)}


class CLIPJoiner(nn.Sequential):
    """Joiner for CLIPViTBackbone — same interface as Joiner."""
    def __init__(self, backbone: CLIPViTBackbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)          # dict {"0": NestedTensor}
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    if args.backbone in CLIPViTBackbone.CONFIGS:
        n_visual_ctx = getattr(args, 'n_visual_ctx', 0) if getattr(args, 'use_visual_prompts', False) else 0
        backbone = CLIPViTBackbone(args.backbone, train_backbone, n_visual_ctx=n_visual_ctx)
        return CLIPJoiner(backbone, position_embedding)
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    return Joiner(backbone, position_embedding)
