# run_suspicion.py
import argparse
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO


def center_xyxy(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def l2(a, b):
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def point_in_poly(p, poly):
    """Return True if point p (x,y) is inside polygon poly (list of [x,y])."""
    if not poly:
        return False
    return cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (int(p[0]), int(p[1])), False) >= 0


def main():
    # ----------------- Args & config -----------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--config", default="configs/suspicion.yaml", help="Path to YAML config")
    ap.add_argument("--out", default="outputs/suspicion_out.mp4", help="Path to save annotated video")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    device = cfg.get("device", "cpu")  # "cpu" or "0" for GPU 0
    yolo_model = cfg.get("yolo_model", "yolov8s.pt")
    conf_thr = float(cfg.get("conf_threshold", 0.25))
    iou_thr = float(cfg.get("iou_threshold", 0.5))

    person_class = cfg.get("person_class", "person")
    item_classes = set(cfg.get("item_classes", ["backpack", "handbag", "suitcase", "book"]))

    # Zones (optional). Leave empty to analyze the full frame.
    shelf_zones = cfg.get("shelf_zones", [])  # list of polygons
    pos_zone = cfg.get("pos_zone", []) or None

    nearby_px = int(cfg.get("nearby_distance_px", 200))
    risk_thr_watch = float(cfg.get("risk_threshold_watch", 0.70))
    risk_thr_action = float(cfg.get("risk_threshold_action", 0.85))
    fps_assume = float(cfg.get("fps_assume", 30))

    # ----------------- Video IO -----------------
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1e-3:
        fps = fps_assume
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path("outputs/events").mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    # ----------------- YOLO -----------------
    model = YOLO(yolo_model)

    # ----------------- Logs -----------------
    frame_idx = 0
    risk_rows = []
    alert_rows = []

    # ----------------- Run -----------------
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t_sec = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

        # Inference
        yres = model.predict(
            source=frame,
            conf=conf_thr,
            iou=iou_thr,
            device=device,
            verbose=False,
        )[0]

        # Split detections
        people = []
        items = []
        for b in yres.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_id = int(b.cls.item())
            cls_name = yres.names[cls_id]
            if cls_name == person_class:
                people.append([x1, y1, x2, y2])
            elif cls_name in item_classes:
                items.append([x1, y1, x2, y2])

        # If no shelf_zones defined, treat whole frame as “watched”
        full_frame_poly = [[0, 0], [W, 0], [W, H], [0, H]]
        zones_to_use = shelf_zones if shelf_zones else [full_frame_poly]

        # Draw zones (if any)
        for poly in zones_to_use:
            poly_np = np.array(poly, dtype=np.int32)
            for i in range(len(poly_np)):
                a = tuple(poly_np[i])
                b = tuple(poly_np[(i + 1) % len(poly_np)])
                cv2.line(frame, a, b, (150, 150, 255), 2)
        if pos_zone:
            pz = np.array(pos_zone, dtype=np.int32)
            for i in range(len(pz)):
                a = tuple(pz[i])
                b = tuple(pz[(i + 1) % len(pz)])
                cv2.line(frame, a, b, (120, 220, 120), 2)

        # Draw items
        for ib in items:
            x1, y1, x2, y2 = map(int, ib)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(
                frame, "item", (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1
            )

        # Per-person risk (simple, per-frame id = index)
        for pid, pb in enumerate(people, start=1):
            cx, cy = center_xyxy(pb)

            # Inside any watched zone?
            in_watched = any(point_in_poly((cx, cy), poly) for poly in zones_to_use)

            # In POS?
            in_pos = bool(pos_zone) and point_in_poly((cx, cy), pos_zone)

            # Near any item?
            near_item = False
            if items:
                p_c = center_xyxy(pb)
                for ib in items:
                    if l2(p_c, center_xyxy(ib)) <= nearby_px:
                        near_item = True
                        break

            # --- Risk formula (simple & robust)
            # Treat being in a watched zone + near an item (and NOT in POS) as suspicious.
            base = 0.0
            if in_watched and near_item and not in_pos:
                base = 0.85  # ACTION-level event
            elif in_watched and near_item:
                base = 0.65  # close to WATCH

            risk = min(1.0, base)

            # Draw person box + risk
            x1, y1, x2, y2 = map(int, pb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"id:{pid} r:{risk:.2f}", (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            # Logs
            risk_rows.append({
                "frame_idx": frame_idx,
                "timestamp_s": t_sec,
                "person_id": pid,
                "risk": risk,
                "in_watched_zone": in_watched,
                "near_item": near_item,
                "in_pos": in_pos,
            })

            # Alerts
            tier = None
            if risk >= risk_thr_action:
                tier = "ACTION"
            elif risk >= risk_thr_watch:
                tier = "WATCH"

            if tier:
                evt = {
                    "event_id": str(uuid.uuid4()),
                    "type": f"RISK_{tier}",
                    "timestamp_s": t_sec,
                    "frame_idx": frame_idx,
                    "person_id": pid,
                    "risk": risk,
                }
                with open(f"outputs/events/{evt['event_id']}.json", "w") as f:
                    json.dump(evt, f, indent=2)
                alert_rows.append(evt)

        # Write frame
        writer.write(frame)
        frame_idx += 1

    # ----------------- Finish -----------------
    cap.release()
    writer.release()

    Path("outputs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(risk_rows).to_csv("outputs/suspicion_risks.csv", index=False)
    pd.DataFrame(alert_rows).to_csv("outputs/suspicion_alerts.csv", index=False)
    print("Saved:", args.out)
    print("Logs: outputs/suspicion_risks.csv, outputs/suspicion_alerts.csv, outputs/events/*.json")


if __name__ == "__main__":
    main()
