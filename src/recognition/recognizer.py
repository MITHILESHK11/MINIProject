# src/recognition/recognizer.py
# Embedding extraction (facenet-pytorch InceptionResnetV1) + DB utilities
import os
import numpy as np
import torch
import cv2
from facenet_pytorch import InceptionResnetV1
from typing import Tuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load pretrained model (vggface2 weights)
_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def _preprocess_face(bgr_face, size=(160,160)):
    # returns torch tensor shape (1,3,H,W) normalized as facenet expects
    img = cv2.resize(bgr_face, size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_norm = (img_rgb / 255.0 - 0.5) / 0.5  # [-1,1]
    tensor = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).to(device).float()
    return tensor

def extract_embedding(bgr_face) -> np.ndarray:
    """Return L2-normalized 512-d embedding as numpy array."""
    with torch.no_grad():
        t = _preprocess_face(bgr_face)
        emb = _model(t)  # (1,512)
        emb = emb.cpu().numpy()[0]
        emb = emb / np.linalg.norm(emb)
        return emb

def build_database(images_root: str = "data/database/images",
                   embeddings_out: str = "data/database/embeddings.npy",
                   meta_out: str = "data/database/meta.npy") -> Tuple[int,int]:
    """
    Walk images_root where each subfolder is a person id:
      data/database/images/<person_id>/*.jpg
    Compute embeddings for every image, save embeddings (N,D) and meta (list of dicts).
    Returns counts (images_processed, unique_persons)
    """
    embeddings = []
    meta = []
    persons = sorted([d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root,d))])
    for person in persons:
        pdir = os.path.join(images_root, person)
        for fname in sorted(os.listdir(pdir)):
            fpath = os.path.join(pdir, fname)
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print("Warning: cannot read", fpath)
                    continue
                emb = extract_embedding(img)
                embeddings.append(emb)
                meta.append({'person': person, 'image': fpath})
            except Exception as e:
                print("Error processing", fpath, e)

    if len(embeddings) == 0:
        raise RuntimeError("No embeddings produced. Check images_root path and images.")
    np.save(embeddings_out, np.stack(embeddings))
    np.save(meta_out, np.array(meta, dtype=object))
    return len(embeddings), len(persons)

def load_db(emb_file: str = "data/database/embeddings.npy",
            meta_file: str = "data/database/meta.npy"):
    emb = np.load(emb_file)
    meta = np.load(meta_file, allow_pickle=True)
    return emb, list(meta)

def match_embedding(query_emb: np.ndarray, db_emb: np.ndarray, topk: int = 5):
    """
    query_emb (D,), db_emb (N,D) both L2-normalized -> cosine similarity = dot product
    Returns (indices, scores)
    """
    sims = np.dot(db_emb, query_emb)  # shape (N,)
    idx = np.argsort(-sims)[:topk]
    return idx, sims[idx]
