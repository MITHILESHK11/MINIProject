# src/detection/detector.py
# Uses facenet-pytorch MTCNN for detection & alignment.
# Install: pip install facenet-pytorch
import torch
from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_mtcnn = MTCNN(keep_all=True, device=device)

def detect_faces(frame_bgr):
    """
    Detect faces in a BGR OpenCV frame.
    Returns list of dicts: {'box': (x1,y1,x2,y2), 'conf': float, 'crop': np.array (BGR)}
    """
    # Convert BGR to RGB PIL
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    boxes, probs = _mtcnn.detect(img)

    faces = []
    if boxes is None:
        return faces

    for box, p in zip(boxes, probs):
        x1, y1, x2, y2 = [int(max(0, v)) for v in box]
        crop = frame_bgr[y1:y2, x1:x2].copy()
        faces.append({'box': (x1, y1, x2, y2), 'conf': float(p), 'crop': crop})
    return faces

def save_aligned(face_crop_bgr, out_path, size=(160,160)):
    """
    Optional helper: resize and save aligned crop.
    """
    img = cv2.resize(face_crop_bgr, size)
    cv2.imwrite(out_path, img)



