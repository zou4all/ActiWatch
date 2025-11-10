# run_behavioral.py
import argparse, json, uuid
from collections import deque, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO


# ---------- Geometry helpers ----------
def center_xyxy(b):
    x1, y1, x2, y2 = b
    return ( (x1 + x2) / 2.0, (y1 + y2) / 2.0 )

def l2(a, b):
    return float(((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5)

def point_in_poly(p, poly):
    if not poly:
        return False
    return cv2.pointPolygonTest(np.array(poly, np.int32), (int(p[0]), int(p[1])), False) >= 0

def angle_deg(a, b, c):
    """Angle ABC (at B) between BA and BC in degrees."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = (v1[0]**2 + v1[1]**2) ** 0.5 + 1e-6
    n2 = (v2[0]**2 + v2[1]**2) ** 0.5 + 1e-6
    cosang = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)
    cosang = max(-1.0, min(1.0, cosang))
    return np.degrees(np.arccos(cosang))


# ---------- Simple IOU tracker (ID persistence) ----------
class Track:
    def __init__(self, tid, bbox):
        self.id = tid
        self.bbox = bbox
        self.age = 0
        self.hits = 1

class IOUTracker:
    def __init__(self, iou_thr=0.2, max_age=20):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0: return 0.0
        ua = (ax2-ax1)*(ay2-ay1); ub = (bx2-bx1)*(by2-by1)
        union = ua + ub - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections):
        # increment age
        for t in self.tracks: t.age += 1

        # match by IOU greedy
        used = set()
        for d in detections:
            best, best_iou = None, 0.0
            for t in self.tracks:
                if t in used: continue
                i = self.iou(t.bbox, d)
                if i > best_iou:
                    best, best_iou = t, i
            if best is not None and best_iou >= self.iou_thr:
                best.bbox = d; best.hits += 1; best.age = 0; used.add(best)
            else:
                nt = Track(self.next_id, d); self.next_id += 1
                self.tracks.append(nt); used.add(nt)

        # drop stale
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return [{"track_id": t.id, "bbox": t.bbox} for t in self.tracks]


# ---------- Face detector (no extra downloads) ----------
def build_face_detector():
    # Haarcascade ships with opencv-python
    face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    fd = cv2.CascadeClassifier(face_xml)
    if fd.empty():
        raise RuntimeError("Failed to load Haar cascade for faces.")
    return fd

def detect_faces_haar(gray_img, roi=None, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    if roi is not None:
        x1, y1, x2, y2 = [int(v) for v in roi]
        sub = gray_img[y1:y2, x1:x2]
        rects = fd.detectMultiScale(sub, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        out = []
        for (x, y, w, h) in rects:
            out.append([x1 + x, y1 + y, x1 + x + w, y1 + y + h])
        return out
    else:
        rects = fd.detectMultiScale(gray_img, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        return [[x, y, x + w, y + h] for (x, y, w, h) in rects]


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="configs/behavioral.yaml")
    ap.add_argument("--out", default="outputs/behavioral_out.mp4")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    yolo_model = cfg.get("yolo_model", "yolov8s.pt")
    attr_model_path = cfg.get("attr_model", "")
    device = cfg.get("device", "cpu")
    conf_thr = float(cfg.get("conf_threshold", 0.25))
    iou_thr  = float(cfg.get("iou_threshold", 0.5))

    person_class = cfg.get("person_class", "person")
    item_classes = set(cfg.get("item_classes", ["backpack", "handbag", "suitcase", "book", "box"]))
    shelf_zones  = cfg.get("shelf_zones", [])
    pos_zone     = cfg.get("pos_zone", []) or None

    near_px = int(cfg.get("nearby_distance_px", 200))
    fps_assume = float(cfg.get("fps_assume", 30))

    # behavioral thresholds
    scan_win = int(cfg.get("face_scan_window", 20))
    scan_thr_deg = float(cfg.get("scan_angle_thr_deg", 25.0))
    min_scan_hits = int(cfg.get("min_scan_hits", 4))

    face_missing_win = int(cfg.get("face_missing_window", 30))
    face_visible_ratio_min = float(cfg.get("face_visible_ratio_min", 0.25))

    risk_thr_watch = float(cfg.get("risk_threshold_watch", 0.65))
    risk_thr_action = float(cfg.get("risk_threshold_action", 0.85))

    # video io
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_assume
    if not fps or fps < 1e-3:
        fps = fps_assume
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    Path("outputs/events").mkdir(parents=True, exist_ok=True)

    # models
    yolo = YOLO(yolo_model)
    attr_model = YOLO(attr_model_path) if (attr_model_path and attr_model_path.strip()) else None
    global fd
    fd = build_face_detector()

    # trackers & per-person memory
    tracker = IOUTracker(iou_thr=0.25, max_age=int(fps))  # ~1s max age
    face_centers = defaultdict(lambda: deque(maxlen=scan_win))   # person_id -> deque of (x,y)
    face_seen_hist = defaultdict(lambda: deque(maxlen=face_missing_win))  # True/False history

    frame_idx = 0
    risk_rows, alert_rows = [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # YOLO infer (people + items)
        res = yolo.predict(source=frame, conf=conf_thr, iou=iou_thr, device=device, verbose=False)[0]
        people = []
        items  = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls = res.names[int(b.cls.item())]
            if cls == person_class:
                people.append([x1, y1, x2, y2])
            elif cls in item_classes:
                items.append([x1, y1, x2, y2])

        # optional attributes model (mask/hat/etc.)
        attr_dets = []
        if attr_model is not None:
            ar = attr_model.predict(source=frame, conf=0.25, iou=0.45, device=device, verbose=False)[0]
            for b in ar.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cls = ar.names[int(b.cls.item())]
                attr_dets.append({"cls": cls, "bbox": [x1, y1, x2, y2]})

        # track persons
        tracks = tracker.update(people)

        # zones to use
        full_poly = [[0, 0], [W, 0], [W, H], [0, H]]
        watched_zones = shelf_zones if shelf_zones else [full_poly]

        # overlays: zones
        for poly in watched_zones:
            p = np.array(poly, np.int32)
            for i in range(len(p)):
                a, b = tuple(p[i]), tuple(p[(i+1) % len(p)])
                cv2.line(frame, a, b, (150,150,255), 2)
        if pos_zone:
            pz = np.array(pos_zone, np.int32)
            for i in range(len(pz)):
                a, b = tuple(pz[i]), tuple(pz[(i+1)%len(pz)])
                cv2.line(frame, a, b, (120,220,120), 2)

        # draw items
        for ib in items:
            x1, y1, x2, y2 = map(int, ib)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)
            cv2.putText(frame, "item", (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)

        # per-person behavioral analysis
        for tr in tracks:
            pid, pb = tr["track_id"], tr["bbox"]
            px, py = center_xyxy(pb)
            in_watched = any(point_in_poly((px, py), z) for z in watched_zones)
            in_pos = bool(pos_zone) and point_in_poly((px, py), pos_zone)

            # nearest item distance
            near_item = False
            if items:
                p_c = (px, py)
                near_item = min(l2(p_c, center_xyxy(ib)) for ib in items) <= near_px

            # face detection within person box
            faces = detect_faces_haar(gray, roi=pb)
            face_visible = len(faces) > 0
            face_seen_hist[pid].append(face_visible)

            # choose the largest face as head proxy
            head_c = None
            if face_visible:
                areas = [(f, (f[2]-f[0])*(f[3]-f[1])) for f in faces]
                fmax = max(areas, key=lambda x: x[1])[0]
                head_c = center_xyxy(fmax)
                face_centers[pid].append(head_c)

            # compute "looking around" from head center angular changes
            scan_hits = 0
            scan_score_deg = 0.0
            fc = face_centers[pid]
            if len(fc) >= 3:
                # sum pairwise angle changes across the window
                angles = []
                for i in range(1, len(fc)-1):
                    ang = angle_deg(fc[i-1], fc[i], fc[i+1])
                    angles.append(ang)
                scan_hits = sum(1 for a in angles if a >= scan_thr_deg)
                scan_score_deg = float(np.mean(angles)) if angles else 0.0

            looking_around = scan_hits >= min_scan_hits

            # face covered or turned away (heuristic): face rarely visible in window while near items
            vis_ratio = (sum(face_seen_hist[pid]) / len(face_seen_hist[pid])) if face_seen_hist[pid] else 0.0
            face_covered_or_turned = (len(face_seen_hist[pid]) >= face_missing_win) and (vis_ratio < cfg.get("face_visible_ratio_min", 0.25)) and near_item

            # attribute-based hints (if attr model provided)
            has_hat = any(d["cls"].lower() in ("hat", "helmet", "hardhat", "cap") and IOUTracker.iou(d["bbox"], pb) > 0.2 for d in attr_dets)
            has_mask = any("mask" in d["cls"].lower() and IOUTracker.iou(d["bbox"], pb) > 0.2 for d in attr_dets)

            # ----- Risk composition -----
            risk = 0.0
            # Base proximity in watched area
            if in_watched and near_item and not in_pos:
                risk = max(risk, 0.70)
            elif in_watched and near_item:
                risk = max(risk, 0.55)

            # Behavioral boosters
            if looking_around:
                risk = min(1.0, risk + 0.15)  # scanning adds suspicion
            if face_covered_or_turned or has_mask:
                risk = min(1.0, risk + 0.20)
            if has_hat:
                risk = min(1.0, risk + 0.05)  # very mild (hats are common)

            tier = None
            if risk >= risk_thr_action:
                tier = "ACTION"
            elif risk >= risk_thr_watch:
                tier = "WATCH"

            # Draw person
            x1,y1,x2,y2 = map(int, pb)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            info = f"id:{pid} r:{risk:.2f}"
            if looking_around: info += " scan"
            if face_covered_or_turned: info += " face_off"
            if has_hat: info += " hat"
            if has_mask: info += " mask"
            cv2.putText(frame, info, (x1, max(15, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Optionally draw head points trail
            for c in list(fc)[-10:]:
                cv2.circle(frame, (int(c[0]), int(c[1])), 2, (255,255,0), -1)

            # Logs
            risk_rows.append({
                "frame_idx": frame_idx, "timestamp_s": t, "person_id": pid,
                "risk": risk,
                "in_watched_zone": in_watched,
                "near_item": near_item,
                "in_pos": in_pos,
                "looking_around": looking_around,
                "face_visible_ratio": round(vis_ratio, 3),
                "face_covered_or_turned": face_covered_or_turned,
                "has_hat": has_hat,
                "has_mask": has_mask
            })

            if tier:
                evt = {
                    "event_id": str(uuid.uuid4()),
                    "type": f"BEHAV_{tier}",
                    "timestamp_s": t,
                    "frame_idx": frame_idx,
                    "person_id": pid,
                    "risk": risk,
                    "looking_around": looking_around,
                    "face_covered_or_turned": face_covered_or_turned,
                    "has_hat": has_hat,
                    "has_mask": has_mask
                }
                with open(f"outputs/events/{evt['event_id']}.json", "w") as f:
                    json.dump(evt, f, indent=2)
                alert_rows.append(evt)

        # write frame and bump
        writer.write(frame)
        frame_idx += 1

    # finalize
    cap.release(); writer.release()
    pd.DataFrame(risk_rows).to_csv("outputs/behavioral_risks.csv", index=False)
    pd.DataFrame(alert_rows).to_csv("outputs/behavioral_alerts.csv", index=False)
    print("Saved:", args.out)
    print("Logs: outputs/behavioral_risks.csv, outputs/behavioral_alerts.csv, outputs/events/*.json")


if __name__ == "__main__":
    main()
