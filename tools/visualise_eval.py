import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import datasets
from util.misc import nested_tensor_from_tensor_list
from models import build_model

# --- Corrected Class List (ID 1-35) ---
VIDVRD_CLASSES = [
    '__background__', 'airplane', 'bicycle', 'bird', 'bus', 'car', 'dog', 'domestic_cat', 
    'elephant', 'hamster', 'lion', 'monkey', 'rabbit', 'sheep', 'snake', 'squirrel', 
    'tiger', 'train', 'turtle', 'whale', 'zebra', 'ball', 'frisbee', 'sofa', 
    'skateboard', 'person', 'horse', 'watercraft', 'giant_panda', 'fox', 'red_panda', 
    'cattle', 'motorcycle', 'bear', 'antelope', 'lizard'
]

# --- Mock Args to trick the dataset builder ---
class MockArgs:
    def __init__(self, args):
        self.dataset_file = 'vidvrd'
        self.vid_path = args.vid_path
        self.masks = False
        self.num_ref_frames = 14 
        self.interval1 = 1 
        self.interval2 = 1
        self.cache_mode = False
        self.obj_split = args.obj_split
        self.eval = True # Force validation mode (no random crop)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def main(args):
    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.set_device(0) 
    device = torch.device(args.device)

    print(f"Loading Model from {args.resume}...")
    model, _, postprocessors = build_model(args)
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print("Loading Dataset...")
    dataset = datasets.build_dataset(image_set='val', args=MockArgs(args))

    for i, idx in enumerate(args.indices):
        print(f"\n[{i+1}/{len(args.indices)}] Processing Index {idx}...")
        
        try:
            img, target = dataset[idx]
        except IndexError:
            print(f"Index {idx} is out of bounds (Max: {len(dataset)-1}). Skipping.")
            continue
            
        image_id = target['image_id'].item()
        
        img = img.to(device)
        samples = [img]
        
        with torch.no_grad():
            outputs = model(samples)

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > args.thresh
        
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), target['orig_size'])
        probas = probas[keep].cpu()
        
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_viz = img[:3].cpu() * std + mean
        img_viz = img_viz.clamp(0, 1).permute(1, 2, 0).numpy()

        plt.figure(figsize=(16, 10))
        plt.imshow(img_viz)
        ax = plt.gca()
        
        colors = plt.cm.hsv(np.linspace(0, 1, len(VIDVRD_CLASSES))).tolist()
        found_objs = 0
        
        for p, (xmin, ymin, xmax, ymax), c in zip(probas, bboxes_scaled.tolist(), colors):
            cl = p.argmax()
            score = p[cl]
            
            if cl < len(VIDVRD_CLASSES):
                text = f'{VIDVRD_CLASSES[cl]}: {score:0.2f}'
                color = colors[cl]
            else:
                text = f'Class {cl}: {score:0.2f}'
                color = 'red'
                
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=color, linewidth=3))
            ax.text(xmin, ymin, text, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
            found_objs += 1

        plt.axis('off')
        plt.title(f"Index {idx} | ID {image_id} | Found {found_objs} objects > {args.thresh}")
        
        save_name = f"viz_idx{idx}.png"
        plt.savefig(save_name)
        plt.close()
        print(f"Saved {save_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_file', default='vidvrd')
    parser.add_argument('--vid_path', required=True, help="Path to dataset root")
    parser.add_argument('--resume', required=True, help="Path to checkpoint.pth")
    parser.add_argument('--indices', type=int, nargs='+', default=[0], help="List of image indices to visualise")
    parser.add_argument('--thresh', type=float, default=0.5, help="Score threshold")
    parser.add_argument('--obj_split', default='base', help="base/novel/all")
    parser.add_argument('--device', default='cuda', help="cuda or cpu")

    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--position_embedding', default='sine', type=str) # Fixes AttributeError
    parser.add_argument('--dilation', default=True, action='store_true')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--frozen_weights', type=str, default=None)

    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--num_feature_levels', default=1, type=int)
    parser.add_argument('--num_ref_frames', default=14, type=int)
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--with_box_refine', default=True, action='store_true')

    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)

    # --- Temporal / Video Module Specifics ---
    parser.add_argument('--n_temporal_decoder_layers', default=1, type=int)
    parser.add_argument('--fixed_pretrained_model', action='store_true')

    # --- 5. Matcher Settings (Required by build_matcher) ---
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)

    args = parser.parse_args()
    main(args)