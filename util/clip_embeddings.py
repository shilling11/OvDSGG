import torch
import clip
import os
import json
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../datasets/vidvrd_dataset'))
from VidVRD_class_split_info import class_split_info

def generate_and_save_clip_embeddings(output_path=os.path.join(os.path.dirname(__file__), '../predicate_embeddings.pt')):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading CLIP model on {device}...")
    model, _ = clip.load("ViT-B/16", device)

    json_path = os.path.join(os.path.dirname(__file__), '../datasets/vidvrd_dataset/VidVRD_pred_class_split_info_v2.json')

    with open(json_path, 'r') as f:
        pred_class_split_info = json.load(f)

    id2cls = pred_class_split_info['id2cls']

    sorted_ids = sorted(int(k) for k, v in id2cls.items() if v != "__background__")
    predicates = [id2cls[str(i)] for i in sorted_ids]
    predicates.insert(0,'background')

    # TODO: Experiment with prompting for text inputs later
    text_inputs = clip.tokenize(predicates).to(device)

    with torch.no_grad():
        text_embeddings = model.encode_text(text_inputs)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    torch.save(text_embeddings.cpu(), output_path)

def generate_and_save_obj_clip_embeddings(output_path=os.path.join(os.path.dirname(__file__), '../object_embeddings.pt')):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading CLIP model on {device}...")
    model, _ = clip.load("ViT-B/16", device)

    id2cls = {v: k for k, v in class_split_info['cls2id'].items()}
    sorted_ids = sorted(k for k, v in id2cls.items() if v != "__background__")
    objects = [id2cls[i] for i in sorted_ids]
    
    objects.insert(0, "background")

    text_inputs = clip.tokenize(objects).to(device)

    with torch.no_grad():
        text_embeddings = model.encode_text(text_inputs)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    torch.save(text_embeddings.cpu(), output_path)
    print(f"Saved {len(objects)} object embeddings to {output_path}")

if __name__ == "__main__":
    generate_and_save_clip_embeddings()
    print(f"Generated predicate embeddings")
    generate_and_save_obj_clip_embeddings()
    print(f"Generated object embeddings")