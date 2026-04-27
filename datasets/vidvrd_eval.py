import copy
import numpy as np
import torch
from collections import defaultdict

from util.misc import all_gather, compute_ap
from util.box_ops import trajectory_iou, box_iou

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_traj_tensors(traj_dict, start_f, t_len, device='cpu'):
    """Convert a {frame_id: [x0,y0,x1,y1]} dict to (1, T, 4) tensor + mask."""
    tensor = torch.zeros((1, t_len, 4), device=device)
    mask   = torch.zeros((1, t_len), dtype=torch.bool, device=device)
    for f, box in traj_dict.items():
        idx = int(f) - start_f
        if 0 <= idx < t_len:
            tensor[0, idx] = torch.tensor(box, device=device)
            mask[0, idx]   = True
    return tensor, mask


def _viou_pred_vs_gts(pred, valid_gts):
    """Compute min(viou_sub, viou_obj) between pred and each valid GT.

    Returns best_viou (float) and best_gt_local_idx (int into valid_gts).
    """
    all_frames = set()
    if isinstance(pred["sub_trajectory"], dict):
        all_frames.update(int(k) for k in pred["sub_trajectory"])
    for gt in valid_gts:
        if isinstance(gt["sub_trajectory"], dict):
            all_frames.update(int(k) for k in gt["sub_trajectory"])

    if not all_frames:
        return 0.0, -1

    start_f, end_f = min(all_frames), max(all_frames)
    t_len = end_f - start_f + 1

    pred_sub, pred_sub_mask = _build_traj_tensors(pred["sub_trajectory"], start_f, t_len)
    pred_obj, pred_obj_mask = _build_traj_tensors(pred["obj_trajectory"], start_f, t_len)

    M = len(valid_gts)
    gt_sub  = torch.zeros((M, t_len, 4))
    gt_obj  = torch.zeros((M, t_len, 4))
    gt_sub_mask = torch.zeros((M, t_len), dtype=torch.bool)
    gt_obj_mask = torch.zeros((M, t_len), dtype=torch.bool)

    for m, gt in enumerate(valid_gts):
        for f, box in gt.get("sub_trajectory", {}).items():
            idx = int(f) - start_f
            if 0 <= idx < t_len:
                gt_sub[m, idx]      = torch.tensor(box)
                gt_sub_mask[m, idx] = True
        for f, box in gt.get("obj_trajectory", {}).items():
            idx = int(f) - start_f
            if 0 <= idx < t_len:
                gt_obj[m, idx]      = torch.tensor(box)
                gt_obj_mask[m, idx] = True

    sub_viou = trajectory_iou(pred_sub, gt_sub, pred_sub_mask, gt_sub_mask)[0]
    obj_viou = trajectory_iou(pred_obj, gt_obj, pred_obj_mask, gt_obj_mask)[0]
    min_viou = torch.min(sub_viou, obj_viou)
    best_viou, best_idx = torch.max(min_viou, dim=0)
    return best_viou.item(), best_idx.item()


def _compute_map_and_recall(preds, ground_truths, viou_threshold,
                            check_sub_label, check_obj_label,
                            ks=(50, 100)):
    """Core PR / mAP / R@K computation shared by SGDet, SGCls, PredCls.

    Args:
        preds:           list of prediction dicts (globally sorted by score descending)
        ground_truths:   dict  vid_id → list of GT dicts (with 'matched' flag)
        viou_threshold:  float
        check_sub_label: bool  — require pred['subject'] == gt['subject']
        check_obj_label: bool  — require pred['object']  == gt['object']
        ks:              tuple of K values for R@K

    Returns:
        dict with 'mAP', 'R@{k}' for each k in ks
    """
    # Reset matched flags
    for gts in ground_truths.values():
        for gt in gts:
            gt["matched"] = False

    npos = sum(len(v) for v in ground_truths.values())
    nd   = len(preds)
    tp   = np.zeros(nd)
    fp   = np.zeros(nd)

    for i, pred in enumerate(preds):
        vid_id = pred["video_id"]
        gts    = ground_truths.get(vid_id, [])

        if not gts:
            fp[i] = 1.0
            continue

        # Filter to unmatched GTs whose labels are consistent with this prediction
        valid_gts     = []
        valid_gt_idxs = []
        for j, gt in enumerate(gts):
            if gt["matched"]:
                continue
            if pred["predicate"] != gt["predicate"]:
                continue
            if check_sub_label and pred["subject"] != gt["subject"]:
                continue
            if check_obj_label and pred["object"] != gt["object"]:
                continue
            valid_gts.append(gt)
            valid_gt_idxs.append(j)

        if not valid_gts:
            fp[i] = 1.0
            continue

        best_viou, best_local = _viou_pred_vs_gts(pred, valid_gts)

        if best_local >= 0 and best_viou >= viou_threshold:
            tp[i] = 1.0
            gts[valid_gt_idxs[best_local]]["matched"] = True
        else:
            fp[i] = 1.0

    fp_cum  = np.cumsum(fp)
    tp_cum  = np.cumsum(tp)
    rec     = tp_cum / float(npos) if npos > 0 else np.zeros(nd)
    prec    = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    ap      = compute_ap(rec, prec)


    # R@K: per-video top-K
    preds_by_vid = defaultdict(list)
    for p in preds:
        preds_by_vid[p["video_id"]].append(p)

    hits_at = {k: 0 for k in ks}
    for k in ks:
        for gts in ground_truths.values():
            for gt in gts:
                gt["matched"] = False

        for vid_id, gts in ground_truths.items():
            vid_preds = sorted(preds_by_vid.get(vid_id, []),
                               key=lambda x: x["score"], reverse=True)[:k]
            for pred in vid_preds:
                valid_gts     = []
                valid_gt_idxs = []
                for j, gt in enumerate(gts):
                    if gt["matched"]:
                        continue
                    if pred["predicate"] != gt["predicate"]:
                        continue
                    if check_sub_label and pred["subject"] != gt["subject"]:
                        continue
                    if check_obj_label and pred["object"] != gt["object"]:
                        continue
                    valid_gts.append(gt)
                    valid_gt_idxs.append(j)

                if not valid_gts:
                    continue

                best_viou, best_local = _viou_pred_vs_gts(pred, valid_gts)
                if best_local >= 0 and best_viou >= viou_threshold:
                    hits_at[k] += 1
                    gts[valid_gt_idxs[best_local]]["matched"] = True

    result = {"mAP": ap}
    for k in ks:
        result[f"R@{k}"] = hits_at[k] / float(npos) if npos > 0 else 0.0
    return result


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class VidVRDEvaluator(object):
    def __init__(self, vidvrd_gt, eval_types, viou_threshold=0.5,
                 pred_split='all', obj_split='all',
                 num_base_pred=None, num_base_obj=None):
        assert isinstance(eval_types, (list, tuple))
        assert pred_split in ('all', 'base', 'novel')
        assert obj_split  in ('all', 'base', 'novel')
        self.eval_types      = eval_types
        self.vidvrd_gt       = copy.deepcopy(vidvrd_gt)
        self.viou_threshold  = viou_threshold
        self.pred_split      = pred_split
        self.obj_split       = obj_split
        self.num_base_pred   = num_base_pred
        self.num_base_obj    = num_base_obj

        self.vid_ids         = []
        self.predictions     = defaultdict(list)
        self.ground_truths   = defaultdict(list)   # vid_id → list of GT dicts

        self._format_ground_truths()

    def _passes_pred_split(self, predicate_label):
        if self.pred_split == 'all' or self.num_base_pred is None:
            return True
        if self.pred_split == 'base':
            return predicate_label <= self.num_base_pred
        # novel
        return predicate_label > self.num_base_pred

    def _passes_obj_split(self, sub_label, obj_label):
        # 'novel' = at least one of sub/obj is a novel class (standard OV-SGG convention)
        # 'base' = both sub and obj are base classes
        if self.obj_split == 'all' or self.num_base_obj is None:
            return True
        if self.obj_split == 'base':
            return sub_label <= self.num_base_obj and obj_label <= self.num_base_obj
        # novel
        return sub_label > self.num_base_obj or obj_label > self.num_base_obj

    def _format_ground_truths(self):
        n_total = 0
        n_kept  = 0
        for vid_id, data in self.vidvrd_gt.items():
            relation_instances = data.get('relation_instances', [])
            trajectories       = data.get('trajectories', {})

            for rel in relation_instances:
                n_total += 1
                pred_label = rel.get('predicate_label', -1)
                sub_label  = rel.get('subject_label',   -1)
                obj_label  = rel.get('object_label',    -1)

                if not self._passes_pred_split(pred_label):
                    continue
                if not self._passes_obj_split(sub_label, obj_label):
                    continue

                sub_tid_str = str(rel.get('subject_tid'))
                obj_tid_str = str(rel.get('object_tid'))

                sub_traj = {str(f): trajectories[sub_tid_str][str(f)]
                            for f in range(rel.get('begin_fid', 0), rel.get('end_fid', 0))
                            if str(f) in trajectories.get(sub_tid_str, {})}
                obj_traj = {str(f): trajectories[obj_tid_str][str(f)]
                            for f in range(rel.get('begin_fid', 0), rel.get('end_fid', 0))
                            if str(f) in trajectories.get(obj_tid_str, {})}

                self.ground_truths[vid_id].append({
                    'subject':        sub_label,
                    'object':         obj_label,
                    'predicate':      pred_label,
                    'sub_trajectory': sub_traj,
                    'obj_trajectory': obj_traj,
                    'matched':        False,
                })
                n_kept += 1

        if self.pred_split != 'all' or self.obj_split != 'all':
            print(f"[VidVRDEvaluator] GT filtered by pred_split='{self.pred_split}', "
                  f"obj_split='{self.obj_split}': kept {n_kept}/{n_total} relations")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, predictions):
        """Update from SGDet-format predictions (vid_id → {sub_trajectories, …})."""
        vid_ids = list(np.unique(list(predictions.keys())))
        self.vid_ids.extend(vid_ids)

        for eval_type in self.eval_types:
            if eval_type in ('sgcls', 'predcls'):
                continue  # handled via update_cls
            prepared = self.prepare(predictions, eval_type)
            for pred in prepared:
                self.predictions[eval_type].append(pred)

    def update_cls(self, predictions, eval_type):
        """Update from SGCls/PredCls postprocessor output.

        predictions: vid_id → {(sub_tid, obj_tid) → {sub_trajectory, obj_trajectory,
                                                       rel_prob_sum, frame_count,
                                                       sub_label, obj_label}}
        """
        assert eval_type in ('sgcls', 'predcls')
        prepared = self._prep_sgcls_predcls(predictions)
        for pred in prepared:
            self.predictions[eval_type].append(pred)

    def synchronize_between_processes(self):
        all_preds = all_gather(self.predictions)
        merged    = defaultdict(list)
        for p_dict in all_preds:
            for eval_type, preds in p_dict.items():
                merged[eval_type].extend(preds)
        self.predictions = merged

    def accumulate(self):
        self.results = {}

        # Summary of predictions vs GT
        n_gt_rels = sum(len(v) for v in self.ground_truths.values())
        pred_counts = {et: len(self.predictions[et]) for et in self.eval_types}
        print(f"\n[Eval] GT: {len(self.ground_truths)} videos, {n_gt_rels} relations | "
              f"Preds: {pred_counts}")

        for eval_type in self.eval_types:
            preds = self.predictions[eval_type]

            if eval_type in ("relation_detection", "sgdet"):
                preds.sort(key=lambda x: x['score'], reverse=True)
                self.results[eval_type] = _compute_map_and_recall(
                    preds, self.ground_truths, self.viou_threshold,
                    check_sub_label=True, check_obj_label=True
                )

            elif eval_type == "sgcls":
                # GT trajectories → viou trivially 1.0; check full label triplet.
                preds.sort(key=lambda x: x['score'], reverse=True)
                self.results[eval_type] = _compute_map_and_recall(
                    preds, self.ground_truths, self.viou_threshold,
                    check_sub_label=True, check_obj_label=True
                )

            elif eval_type == "predcls":
                # GT trajectories + GT labels → only predicate needs to match.
                preds.sort(key=lambda x: x['score'], reverse=True)
                self.results[eval_type] = _compute_map_and_recall(
                    preds, self.ground_truths, self.viou_threshold,
                    check_sub_label=False, check_obj_label=False
                )

            elif eval_type == "relation_tagging":
                p_at_1, p_at_5, p_at_10 = [], [], []
                preds_by_vid = defaultdict(list)
                for p in preds:
                    preds_by_vid[p["video_id"]].append(p)

                for vid_id in self.ground_truths.keys():
                    vid_preds = sorted(preds_by_vid.get(vid_id, []),
                                       key=lambda x: x['score'], reverse=True)

                    gt_tags = set()
                    for gt in self.ground_truths[vid_id]:
                        gt_tags.add((gt["subject"], gt["predicate"], gt["object"]))

                    hits = np.array([1 if (p["subject"], p["predicate"], p["object"]) in gt_tags
                                     else 0
                                     for p in vid_preds])

                    p_at_1.append(np.sum(hits[:1])  / 1.0  if len(hits) >= 1 else 0.0)
                    p_at_5.append(np.sum(hits[:5])  / 5.0  if len(hits) >= 1 else 0.0)
                    p_at_10.append(np.sum(hits[:10]) / 10.0 if len(hits) >= 1 else 0.0)

                self.results[eval_type] = {
                    "P@1":  np.mean(p_at_1)  if p_at_1  else 0.0,
                    "P@5":  np.mean(p_at_5)  if p_at_5  else 0.0,
                    "P@10": np.mean(p_at_10) if p_at_10 else 0.0,
                }

            elif eval_type == 'object_trajectory':
                preds_by_vid = defaultdict(list)
                for p in preds:
                    preds_by_vid[p['video_id']].append(p)

                total_gt       = 0
                total_hit_at_5  = 0
                total_hit_at_10 = 0

                for vid_id, data in self.vidvrd_gt.items():
                    gt_objs        = []
                    seen_tids      = set()
                    trajectories   = data.get('trajectories', {})

                    for rel in data.get('relation_instances', []):
                        for prefix, tid_key in [('subject', 'subject_tid'), ('object', 'object_tid')]:
                            tid   = str(rel.get(tid_key))
                            label = rel.get(f'{prefix}_label')
                            if tid not in seen_tids and tid in trajectories:
                                seen_tids.add(tid)
                                frames = [int(f) for f in trajectories[tid].keys()]
                                if frames:
                                    gt_objs.append({
                                        'label':      label,
                                        'trajectory': trajectories[tid],
                                        'matched':    False,
                                    })

                    total_gt += len(gt_objs)

                    vid_preds = sorted(preds_by_vid.get(vid_id, []),
                                       key=lambda x: x['score'], reverse=True)

                    for k in [5, 10]:
                        topk_preds = vid_preds[:k]
                        hits = 0
                        for gt in gt_objs:
                            gt['matched'] = False

                        for pred in topk_preds:
                            pred_frames = [int(f) for f in pred['trajectory'].keys()]
                            if not pred_frames:
                                continue

                            best_viou   = 0.0
                            best_gt_idx = -1

                            for j, gt in enumerate(gt_objs):
                                if not gt['matched'] and pred['label'] == gt['label']:
                                    gt_frames  = [int(f) for f in gt['trajectory'].keys()]
                                    all_frames = set(pred_frames) | set(gt_frames)
                                    start_f, end_f = min(all_frames), max(all_frames)
                                    t_len = end_f - start_f + 1

                                    pred_t, pred_m = _build_traj_tensors(pred['trajectory'], start_f, t_len)
                                    gt_t,   gt_m   = _build_traj_tensors(gt['trajectory'],   start_f, t_len)

                                    viou = trajectory_iou(pred_t, gt_t, pred_m, gt_m)[0].item()
                                    if viou > best_viou:
                                        best_viou   = viou
                                        best_gt_idx = j

                            if best_gt_idx >= 0 and best_viou >= self.viou_threshold:
                                hits += 1
                                gt_objs[best_gt_idx]['matched'] = True

                        if k == 5:
                            total_hit_at_5  += hits
                        elif k == 10:
                            total_hit_at_10 += hits

                self.results[eval_type] = {
                    "R@5":  total_hit_at_5  / float(total_gt) if total_gt > 0 else 0.0,
                    "R@10": total_hit_at_10 / float(total_gt) if total_gt > 0 else 0.0,
                }

    def summarize(self):
        for eval_type, metrics in self.results.items():
            print(f"--- {eval_type.upper()} ---")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

    # ------------------------------------------------------------------
    # Prepare: convert raw postprocessor output → flat prediction dicts
    # ------------------------------------------------------------------

    def prepare(self, predictions, eval_type):
        if eval_type in ('relation_detection', 'sgdet'):
            return self._prep_rel_det(predictions)
        elif eval_type == 'relation_tagging':
            return self._prep_rel_tagging(predictions)
        elif eval_type == 'object_trajectory':
            return self._prep_obj_traj(predictions)
        else:
            raise ValueError(f"Unknown eval type: {eval_type}")

    def _prep_rel_det(self, predictions):
        results = []
        for vid_id, pred in predictions.items():
            if len(pred) == 0:
                continue
            scores      = pred['scores'].tolist()
            sub_labels  = pred['sub_labels'].tolist()
            obj_labels  = pred['obj_labels'].tolist()
            predicates  = pred['predicates'].tolist()
            sub_trajs   = pred['sub_trajectories']
            obj_trajs   = pred['obj_trajectories']

            results.extend([
                {
                    'video_id':       vid_id,
                    'subject':        sub_labels[k],
                    'object':         obj_labels[k],
                    'predicate':      predicates[k],
                    'score':          scores[k],
                    'sub_trajectory': sub_traj,
                    'obj_trajectory': obj_traj,
                }
                for k, (sub_traj, obj_traj) in enumerate(zip(sub_trajs, obj_trajs))
                if scores[k] >= 0.01
            ])
        return results

    def _prep_rel_tagging(self, predictions):
        results = []
        for vid_id, pred in predictions.items():
            if len(pred) == 0:
                continue
            scores     = pred['scores'].tolist()
            sub_labels = pred['sub_labels'].tolist()
            obj_labels = pred['obj_labels'].tolist()
            predicates = pred['predicates'].tolist()

            results.extend([
                {
                    'video_id':  vid_id,
                    'subject':   sub_labels[k],
                    'object':    obj_labels[k],
                    'predicate': predicates[k],
                    'score':     scores[k],
                }
                for k in range(len(scores))
            ])
        return results

    def _prep_obj_traj(self, predictions):
        results = []
        for vid_id, pred in predictions.items():
            if len(pred) == 0:
                continue
            trajs  = pred.get('trajectories', pred.get('sub_trajectories', []))
            scores = pred.get('scores', pred.get('sub_scores', [])).tolist()
            labels = pred.get('labels', pred.get('sub_labels', [])).tolist()

            results.extend([
                {
                    'video_id':   vid_id,
                    'label':      labels[k],
                    'score':      scores[k],
                    'trajectory': trajs[k],
                }
                for k in range(len(scores))
            ])
        return results

    def _prep_sgcls_predcls(self, predictions):
        """Flatten the (vid_id → {(s_tid, o_tid) → data}) structure from SGCls/PredCls
        postprocessors into the standard flat prediction list format.

        For each GT pair we emit the top-K predicate classes (ranked by model
        per-class probability averaged over frames). This matches the standard
        SGG protocol for "unconstrained" PredCls and avoids the per-pair tail
        of near-zero predictions inflating R@K while polluting mAP.
        """
        TOP_K_PER_PAIR = 10
        results = []
        for vid_id, pair_dict in predictions.items():
            for (s_tid, o_tid, _gt_pred), data in pair_dict.items():
                n_frames = max(data['frame_count'], 1)
                # Average probability across frames
                rel_prob = data['rel_prob_sum'] / n_frames   # (num_pred_cls,)

                sub_traj = data['sub_trajectory']
                obj_traj = data['obj_trajectory']
                sub_lbl  = data['sub_label']
                obj_lbl  = data['obj_label']

                # Top-K predicate classes per pair
                k_eff = min(TOP_K_PER_PAIR, rel_prob.shape[0])
                top_scores, top_indices = rel_prob.topk(k_eff)
                for score, pred_cls in zip(top_scores.tolist(), top_indices.tolist()):
                    if score < 0.01:
                        continue
                    results.append({
                        'video_id':       vid_id,
                        'subject':        sub_lbl,
                        'object':         obj_lbl,
                        'predicate':      pred_cls,
                        'score':          score,
                        'sub_trajectory': sub_traj,
                        'obj_trajectory': obj_traj,
                    })
        return results
