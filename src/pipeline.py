from src.detection.detector import detect_faces
from src.recognition.recognizer import extract_embedding, match_embedding
from src.aging.aging_model import apply_aging

def run_pipeline(image):
    faces = detect_faces(image)
    embeddings = [extract_embedding(f) for f in faces]
    # TODO: match embeddings with database
    return embeddings
