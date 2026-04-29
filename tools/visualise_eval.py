"""
Qualitative visualiser for PRISM eval predictions.

Loads backup_eval_preds.pth (saved by engine_multi.py with --save_preds),
then renders object boxes and relation triplets onto extracted video frames.

Usage:
    python tools/visualise_eval.py \
        --preds   exps/clip_vitb16/stage3/backup_eval_preds.pth \
        --frames  data/vidvrd/vidvrd_extracted_frames/val \
        --out_dir qualitative/ \
        --video   ILSVRC2015_train_00010001 \
        --top_k   5 \
        --score_thresh 0.05

    # List available video IDs in the preds file:
    python tools/visualise_eval.py --preds <path> --list_videos
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Class name maps
# ---------------------------------------------------------------------------

OBJ_ID2CLS = {
    0: '__background__', 1: 'airplane', 2: 'bicycle', 3: 'bird', 4: 'bus',
    5: 'car', 6: 'dog', 7: 'domestic_cat', 8: 'elephant', 9: 'hamster',
    10: 'lion', 11: 'monkey', 12: 'rabbit', 13: 'sheep', 14: 'snake',
    15: 'squirrel', 16: 'tiger', 17: 'train', 18: 'turtle', 19: 'whale',
    20: 'zebra', 21: 'ball', 22: 'frisbee', 23: 'sofa', 24: 'skateboard',
    25: 'person', 26: 'horse', 27: 'watercraft', 28: 'giant_panda', 29: 'fox',
    30: 'red_panda', 31: 'cattle', 32: 'motorcycle', 33: 'bear',
    34: 'antelope', 35: 'lizard',
}

def _load_pred_id2cls():
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'datasets',
                     'vidvrd_dataset', 'VidVRD_pred_class_split_info_v2.json'),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.exists(p):
            with open(p) as f:
                return {int(k): v for k, v in json.load(f)['id2cls'].items()}
    return {}

PRED_ID2CLS = _load_pred_id2cls()

def obj_name(label):
    return OBJ_ID2CLS.get(int(label), f'cls{label}')

def pred_name(label):
    return PRED_ID2CLS.get(int(label), f'pred{label}')

# ---------------------------------------------------------------------------
# Colour palette — one stable colour per object class
# ---------------------------------------------------------------------------

_PALETTE = plt.cm.get_cmap('tab20', 36).colors

def obj_colour(label):
    return _PALETTE[int(label) % len(_PALETTE)]

# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def traj_box_at_frame(traj, frame_id):
    """Return [x0,y0,x1,y1] for a given frame_id, or None."""
    fid = str(frame_id)
    if isinstance(traj, dict):
        if fid in traj:
            return traj[fid]
        # try int key
        if frame_id in traj:
            return traj[frame_id]
    return None

def traj_centre(box):
    """Centre point of an [x0,y0,x1,y1] box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

# ---------------------------------------------------------------------------
# Core visualisation for a single frame
# ---------------------------------------------------------------------------

def visualise_frame(frame_path, frame_id, relations, score_thresh, top_k,
                    out_path, title=''):
    if not os.path.exists(frame_path):
        print(f'  [skip] frame not found: {frame_path}')
        return

    img = np.array(Image.open(frame_path).convert('RGB'))
    h, w = img.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(img)
    ax.axis('off')

    # Filter to relations visible in this frame and above threshold
    visible = []
    for rel in relations:
        if rel['score'] < score_thresh:
            continue
        sb = traj_box_at_frame(rel['sub_trajectory'], frame_id)
        ob = traj_box_at_frame(rel['obj_trajectory'], frame_id)
        if sb is not None and ob is not None:
            visible.append((rel, sb, ob))

    # Sort by score, take top_k
    visible.sort(key=lambda x: x[0]['score'], reverse=True)
    visible = visible[:top_k]

    drawn_boxes = {}  # box_key → colour, to avoid duplicate box draws

    for rel, sb, ob in visible:
        sub_lbl = rel['subject']
        obj_lbl = rel['object']
        pred_lbl = rel['predicate']
        score = rel['score']

        sc = obj_colour(sub_lbl)
        oc = obj_colour(obj_lbl)

        # Draw subject box
        sb_key = tuple(map(int, sb))
        if sb_key not in drawn_boxes:
            rect = mpatches.FancyBboxPatch(
                (sb[0], sb[1]), sb[2] - sb[0], sb[3] - sb[1],
                boxstyle='round,pad=2', linewidth=2.5,
                edgecolor=sc, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(sb[0], sb[1] - 4, obj_name(sub_lbl),
                    fontsize=9, color='white', fontweight='bold',
                    bbox=dict(facecolor=sc, alpha=0.8, pad=1, edgecolor='none'))
            drawn_boxes[sb_key] = sc

        # Draw object box
        ob_key = tuple(map(int, ob))
        if ob_key not in drawn_boxes:
            rect = mpatches.FancyBboxPatch(
                (ob[0], ob[1]), ob[2] - ob[0], ob[3] - ob[1],
                boxstyle='round,pad=2', linewidth=2.5,
                edgecolor=oc, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(ob[0], ob[1] - 4, obj_name(obj_lbl),
                    fontsize=9, color='white', fontweight='bold',
                    bbox=dict(facecolor=oc, alpha=0.8, pad=1, edgecolor='none'))
            drawn_boxes[ob_key] = oc

        # Draw arrow from subject centre to object centre
        sc_pt = traj_centre(sb)
        oc_pt = traj_centre(ob)
        ax.annotate(
            '', xy=oc_pt, xytext=sc_pt,
            arrowprops=dict(arrowstyle='->', color='yellow', lw=2.0)
        )

        # Predicate label at midpoint
        mid_x = (sc_pt[0] + oc_pt[0]) / 2
        mid_y = (sc_pt[1] + oc_pt[1]) / 2
        label = f'{pred_name(pred_lbl)}\n({score:.2f})'
        ax.text(mid_x, mid_y, label, fontsize=8, color='black', ha='center',
                bbox=dict(facecolor='yellow', alpha=0.75, pad=2, edgecolor='none'))

    n = len(visible)
    ax.set_title(
        title or f'Frame {frame_id} — {n} relation{"s" if n != 1 else ""} shown',
        fontsize=12, pad=8
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out_path}')

# ---------------------------------------------------------------------------
# Multi-frame strip for a video
# ---------------------------------------------------------------------------

def visualise_video_strip(video_id, relations, frames_root, out_dir,
                          score_thresh, top_k, n_frames=6):
    """Pick n_frames evenly spaced frames that have visible predictions."""
    frame_dir = os.path.join(frames_root, video_id)
    if not os.path.isdir(frame_dir):
        print(f'[warn] frame directory not found: {frame_dir}')
        return

    all_frames = sorted(
        int(f.stem) for f in Path(frame_dir).glob('*.jpg') if f.stem.isdigit()
    )
    if not all_frames:
        all_frames = sorted(
            int(f.stem) for f in Path(frame_dir).glob('*.png') if f.stem.isdigit()
        )
    if not all_frames:
        print(f'[warn] no frames found in {frame_dir}')
        return

    # Collect frames that have at least one visible relation
    def has_visible(fid):
        for rel in relations:
            if rel['score'] < score_thresh:
                continue
            if (traj_box_at_frame(rel['sub_trajectory'], fid) is not None and
                    traj_box_at_frame(rel['obj_trajectory'], fid) is not None):
                return True
        return False

    visible_frames = [f for f in all_frames if has_visible(f)]
    if not visible_frames:
        print(f'[warn] no frames with visible predictions for {video_id}')
        return

    # Pick n_frames evenly spaced
    indices = np.round(np.linspace(0, len(visible_frames) - 1, n_frames)).astype(int)
    chosen = [visible_frames[i] for i in indices]

    # Individual frame saves
    for fid in chosen:
        fname = f'{fid:06d}.jpg'
        frame_path = os.path.join(frame_dir, fname)
        if not os.path.exists(frame_path):
            fname = f'{fid:06d}.png'
            frame_path = os.path.join(frame_dir, fname)
        out_path = os.path.join(out_dir, video_id, f'frame_{fid:06d}.png')
        visualise_frame(frame_path, fid, relations, score_thresh, top_k,
                        out_path, title=f'{video_id} — frame {fid}')

    # Composite strip
    _make_strip(video_id, chosen, frame_dir, relations, score_thresh,
                top_k, out_dir)

def _make_strip(video_id, frame_ids, frame_dir, relations, score_thresh,
                top_k, out_dir):
    n = len(frame_ids)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, fid in zip(axes, frame_ids):
        fname = f'{fid:06d}.jpg'
        frame_path = os.path.join(frame_dir, fname)
        if not os.path.exists(frame_path):
            frame_path = os.path.join(frame_dir, f'{fid:06d}.png')

        if not os.path.exists(frame_path):
            ax.axis('off')
            continue

        img = np.array(Image.open(frame_path).convert('RGB'))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'frame {fid}', fontsize=8)

        visible = []
        for rel in relations:
            if rel['score'] < score_thresh:
                continue
            sb = traj_box_at_frame(rel['sub_trajectory'], fid)
            ob = traj_box_at_frame(rel['obj_trajectory'], fid)
            if sb and ob:
                visible.append((rel, sb, ob))
        visible.sort(key=lambda x: x[0]['score'], reverse=True)
        visible = visible[:top_k]

        drawn = {}
        for rel, sb, ob in visible:
            sc = obj_colour(rel['subject'])
            oc = obj_colour(rel['object'])

            for box, col, lbl in [(sb, sc, rel['subject']), (ob, oc, rel['object'])]:
                bk = tuple(map(int, box))
                if bk not in drawn:
                    rect = mpatches.FancyBboxPatch(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        boxstyle='round,pad=1', linewidth=1.5,
                        edgecolor=col, facecolor='none'
                    )
                    ax.add_patch(rect)
                    ax.text(box[0], box[1] - 3, obj_name(lbl),
                            fontsize=6, color='white', fontweight='bold',
                            bbox=dict(facecolor=col, alpha=0.8, pad=0.5, edgecolor='none'))
                    drawn[bk] = col

            sc_pt = traj_centre(sb)
            oc_pt = traj_centre(ob)
            ax.annotate('', xy=oc_pt, xytext=sc_pt,
                        arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))
            mid_x = (sc_pt[0] + oc_pt[0]) / 2
            mid_y = (sc_pt[1] + oc_pt[1]) / 2
            ax.text(mid_x, mid_y, pred_name(rel['predicate']),
                    fontsize=5.5, color='black', ha='center',
                    bbox=dict(facecolor='yellow', alpha=0.7, pad=1, edgecolor='none'))

    fig.suptitle(video_id, fontsize=10, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{video_id}_strip.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  strip saved → {out_path}')

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_id2name(anno_path):
    """Build {int_id: folder_name} from instances_val.json."""
    if not anno_path or not os.path.exists(anno_path):
        return {}
    with open(anno_path) as f:
        d = json.load(f)
    videos = d.get('videos', d.get('video', []))
    return {v['id']: v['name'] for v in videos}


def main():
    parser = argparse.ArgumentParser(description='PRISM qualitative visualiser')
    parser.add_argument('--preds', required=True,
                        help='Path to backup_eval_preds.pth')
    parser.add_argument('--frames', required=True,
                        help='Root of extracted frames: <root>/<video_name>/<frame>.jpg')
    parser.add_argument('--anno', default=None,
                        help='Path to instances_val.json (needed to map int IDs → folder names). '
                             'Default: auto-detected from <frames>/../../annotations/instances_val.json')
    parser.add_argument('--out_dir', default='qualitative',
                        help='Output directory')
    parser.add_argument('--video', nargs='+', default=None,
                        help='Integer video ID(s) or folder name(s) to visualise. '
                             'If omitted, picks top-5 by pred count.')
    parser.add_argument('--score_thresh', type=float, default=0.05,
                        help='Minimum relation score to display')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Max relations to show per frame')
    parser.add_argument('--n_frames', type=int, default=6,
                        help='Number of frames per video strip')
    parser.add_argument('--list_videos', action='store_true',
                        help='Print available video IDs and names, then exit')
    args = parser.parse_args()

    import torch
    from collections import defaultdict

    # --- Build int ID → folder name mapping ---
    anno_path = args.anno
    if anno_path is None:
        # Try to auto-detect relative to frames root
        candidate = os.path.join(args.frames, '..', '..', 'annotations', 'instances_val.json')
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            anno_path = candidate
            print(f'Auto-detected annotations: {anno_path}')
    id2name = _build_id2name(anno_path)
    if not id2name:
        print('[warn] Could not load annotations — will use raw video_id as folder name.')

    def resolve_name(vid_id):
        """Convert integer or string video_id to the frame folder name."""
        try:
            int_id = int(vid_id)
            return id2name.get(int_id, str(vid_id))
        except (ValueError, TypeError):
            return str(vid_id)

    # --- Load predictions ---
    print(f'Loading predictions from {args.preds} ...')
    preds = torch.load(args.preds, map_location='cpu', weights_only=False)

    rel_preds = None
    for key in ('sgdet', 'relation_detection'):
        if key in preds and preds[key]:
            rel_preds = preds[key]
            print(f'Using eval type: {key} ({len(rel_preds)} predictions)')
            break

    if rel_preds is None:
        print('Available eval types:', list(preds.keys()))
        print('No sgdet/relation_detection predictions found.')
        sys.exit(1)

    # Group by video_id (raw, as stored in preds)
    by_video = defaultdict(list)
    for p in rel_preds:
        by_video[p['video_id']].append(p)

    if args.list_videos:
        print(f'\n{"ID":<6} {"Folder name":<40} {"# preds":>8}')
        print('-' * 58)
        for vid, ps in sorted(by_video.items(), key=lambda x: -len(x[1])):
            name = resolve_name(vid)
            print(f'{str(vid):<6} {name:<40} {len(ps):>8}')
        return

    # --- Resolve which videos to visualise ---
    if args.video:
        # User may pass either int IDs or folder names
        # Build reverse map name→id for lookup
        name2id = {v: k for k, v in id2name.items()}
        resolved = []
        for v in args.video:
            # Try as int ID directly
            try:
                int_id = int(v)
                if int_id in by_video:
                    resolved.append(int_id)
                    continue
            except ValueError:
                pass
            # Try as folder name → look up int id
            if v in name2id and name2id[v] in by_video:
                resolved.append(name2id[v])
            else:
                print(f'[warn] {v} not found in predictions — skipping')
        chosen_vids = resolved
    else:
        chosen_vids = [v for v, _ in
                       sorted(by_video.items(), key=lambda x: -len(x[1]))[:5]]
        print(f'No --video specified. Visualising top-5 by pred count:')
        for v in chosen_vids:
            print(f'  {v} → {resolve_name(v)}')

    os.makedirs(args.out_dir, exist_ok=True)

    for vid in chosen_vids:
        folder_name = resolve_name(vid)
        relations = by_video[vid]
        print(f'\n{vid} ({folder_name}): {len(relations)} predictions')
        visualise_video_strip(
            folder_name, relations, args.frames, args.out_dir,
            args.score_thresh, args.top_k, args.n_frames
        )

    print('\nDone.')

if __name__ == '__main__':
    main()
