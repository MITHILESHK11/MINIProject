# src/pipeline.py
# End-to-end pipeline (detect -> embed -> match)
import cv2
import time
from src.detection.detector import detect_faces
from src.recognition.recognizer import extract_embedding, load_db, match_embedding

def process_frame(frame_bgr, db_emb, db_meta, topk=5, conf_threshold=0.90):
    results = []
    faces = detect_faces(frame_bgr)
    for f in faces:
        if f['conf'] < conf_threshold:
            continue
        emb = extract_embedding(f['crop'])
        idxs, scores = match_embedding(emb, db_emb, topk=topk)
        candidate_list = []
        for i, s in zip(idxs, scores):
            meta = db_meta[i]
            candidate_list.append({'person': meta['person'], 'image': meta['image'], 'score': float(s)})
        results.append({
            'box': f['box'],
            'conf': f['conf'],
            'candidates': candidate_list
        })
    return results

def run_video_matches(video_path, topk=5, display=False, sample_rate=5, conf_threshold=0.90):
    db_emb, db_meta = load_db()
    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        # sample frames to reduce load
        if frame_no % sample_rate != 0:
            continue
        results = process_frame(frame, db_emb, db_meta, topk=topk, conf_threshold=conf_threshold)
        for res in results:
            print("Detected face at", res['box'], "candidates:")
            for c in res['candidates']:
                print("  -", c['person'], f"(score={c['score']:.3f}) image={c['image']}")
        if display:
            # draw boxes & top candidate name on frame for quick visual debug
            for res in results:
                x1,y1,x2,y2 = res['box']
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                if res['candidates']:
                    name = res['candidates'][0]['person']
                    score = res['candidates'][0]['score']
                    cv2.putText(frame, f"{name} {score:.2f}", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("minipp debug", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    if display:
        cv2.destroyAllWindows()
    print("Finished. Processed frames with sampling:", sample_rate)
