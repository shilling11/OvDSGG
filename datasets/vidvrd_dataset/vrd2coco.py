import json
import os
import glob
import numpy as np
from tqdm import tqdm
from VidVRD_class_split_info import class_split_info

DATA_ROOT = "../../data/vidvrd"

TRAIN_ANNO_DIR = os.path.join(DATA_ROOT, "annotations/train_anno") 
VAL_ANNO_DIR = os.path.join(DATA_ROOT, "annotations/test_anno")

OUTPUT_TRAIN_JSON = os.path.join(DATA_ROOT, "annotations/instances_train_base.json")
OUTPUT_VAL_JSON = os.path.join(DATA_ROOT, "annotations/instances_val.json")

def convert_vidvrd_to_coco(anno_dir, output_file, is_train=True, num_ref_frames=14):
    print(f"Processing {'TRAIN (Base Only)' if is_train else 'VAL (All Classes)'} from {anno_dir}...")
    
    json_files = glob.glob(os.path.join(anno_dir, "*.json"))
    if not json_files:
        print(f"Warning: No JSON files found in {anno_dir}")
        return

    coco_output = {
        "info": {"description": "VidVRD to COCO conversion with Relations"},
        "categories": [],
        "predicate_categories": [],
        "videos": [],
        "images": [],
        "annotations": []
    }

    with open('VidVRD_pred_class_split_info_v2.json', 'r') as f:
        pred_class_split_info = json.load(f)

    for name, cid in class_split_info["cls2id"].items():
        if name == "__background__": continue
        coco_output["categories"].append({"id": cid, "name": name})

    for name, cid in pred_class_split_info["cls2id"].items():
        if name == "__background__": continue
        coco_output["predicate_categories"].append({"id": cid, "name": name})

    global_vid_id = 1
    global_img_id = 1
    global_ann_id = 1
    
    target_frames_count = num_ref_frames + 1

    for json_file in tqdm(json_files):
        with open(json_file, 'r') as f:
            vid_data = json.load(f)

        video_name = vid_data['video_id']
        
        tid_to_category = {}
        for subj_obj in vid_data['subject/objects']:
            tid_to_category[subj_obj['tid']] = subj_obj['category']

        trajectories = vid_data['trajectories']
        relation_instances = vid_data.get('relation_instances')
        num_frames = vid_data.get('frame_count')
        
        if is_train:
            if num_frames >= target_frames_count:
                frames_to_process = np.linspace(0, num_frames - 1, target_frames_count, dtype=int).tolist()
            else:
                frames_to_process = list(range(num_frames))
            vid_train_frames = frames_to_process.copy()
        else:
            frames_to_process = list(range(num_frames))
            vid_train_frames = []

        vid_rel_instances = []
        for rel in relation_instances:
            pred_name = rel['predicate']
            if pred_name in pred_class_split_info["cls2id"]:
                new_rel = rel.copy()
                new_rel['predicate_label'] = pred_class_split_info["cls2id"][pred_name]
                
                sub_tid = new_rel['subject_tid']
                obj_tid = new_rel['object_tid']
                if sub_tid in tid_to_category and obj_tid in tid_to_category:
                    new_rel['subject_label'] = class_split_info["cls2id"][tid_to_category[sub_tid]]
                    new_rel['object_label'] = class_split_info["cls2id"][tid_to_category[obj_tid]]
                
                del new_rel['predicate']
                vid_rel_instances.append(new_rel)

        vid_trajectories = {}
        for fid, frame in enumerate(trajectories):
            for box in frame:
                tid = box.get('tid')
                if tid not in vid_trajectories:
                    vid_trajectories[tid] = {}
                vid_trajectories[tid][str(fid)] = [box['bbox']['xmin'], box['bbox']['ymin'], box['bbox']['xmax'], box['bbox']['ymax']]
        
        vid_entry = {
            "id": global_vid_id,
            "name": video_name,
            "width": vid_data['width'],
            "height": vid_data['height'],
            "fps": vid_data.get('fps', 30),
            "frame_count": vid_data['frame_count'],
            "vid_train_frames": vid_train_frames,
            "trajectories": vid_trajectories,
            "relation_instances": vid_rel_instances
        }
        
        for frame_idx in frames_to_process:
            file_name = f"{video_name}/{frame_idx:06d}.jpg"
            image_entry = {
                "id": global_img_id,
                "file_name": file_name,
                "height": vid_data['height'],
                "width": vid_data['width'],
                "frame_id": frame_idx,
                "video_id": global_vid_id,
                "is_vid_train_frame": is_train,
            }
            coco_output["images"].append(image_entry)

            frame_objects = trajectories[frame_idx] if frame_idx < len(trajectories) else []
            for obj in frame_objects:
                tid = obj['tid']
                if tid not in tid_to_category: continue 

                cat_name = tid_to_category[tid]
                
                if is_train:
                    split_type = class_split_info["cls2split"].get(cat_name)
                    if split_type == "novel": continue

                if cat_name not in class_split_info["cls2id"]: continue

                category_id = class_split_info["cls2id"][cat_name]

                bbox_raw = obj['bbox']
                xmin, ymin = bbox_raw['xmin'], bbox_raw['ymin']
                xmax, ymax = bbox_raw['xmax'], bbox_raw['ymax']
                w, h = xmax - xmin, ymax - ymin
                
                ann_entry = {
                    "id": global_ann_id,
                    "image_id": global_img_id,
                    "video_id": global_vid_id,
                    "category_id": category_id,
                    "instance_id": int(tid),
                    "bbox": [xmin, ymin, w, h],
                    "area": w * h,
                    "iscrowd": False,
                    "occluded": False,
                    "generated": bool(obj.get("generated", 0))
                }
                
                coco_output["annotations"].append(ann_entry)
                global_ann_id += 1

            global_img_id += 1
        
        coco_output["videos"].append(vid_entry)
        global_vid_id += 1

    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(coco_output, f)
    
    print(f"Done. Saved {len(coco_output['images'])} images and {len(coco_output['annotations'])} annotations.")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_TRAIN_JSON), exist_ok=True)
    convert_vidvrd_to_coco(TRAIN_ANNO_DIR, OUTPUT_TRAIN_JSON, is_train=True, num_ref_frames=14)
    convert_vidvrd_to_coco(VAL_ANNO_DIR, OUTPUT_VAL_JSON, is_train=False, num_ref_frames=14)
