import argparse, yaml, cv2, numpy as np, pandas as pd, time, math
from pathlib import Path
from ultralytics import YOLO
from collections import deque
def bbox_center_xyxy(b): x1,y1,x2,y2=b; return ((x1+x2)/2.0,(y1+y2)/2.0)
def l2(a,b): import math; return math.hypot(a[0]-b[0], a[1]-b[1])
def roi_presence_score(gray, poly):
    mask=np.zeros(gray.shape,np.uint8); poly_np=np.array(poly,np.int32); cv2.fillPoly(mask,[poly_np],255)
    vals=gray[mask==255]; return float(vals.mean()) if vals.size>0 else 0.0
def run_pipeline(video_path, out_path, cfg):
    cap=cv2.VideoCapture(video_path); fps=cap.get(cv2.CAP_PROP_FPS) or float(cfg.get('fps_assume',25)); fps=fps if fps>1e-3 else float(cfg.get('fps_assume',25))
    W,H=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))
    model=YOLO(cfg.get('yolo_model','yolov8s.pt')); conf=cfg.get('conf_threshold',0.25); iou=cfg.get('iou_threshold',0.5); device=cfg.get('device','cpu')
    pkg_classes=set(cfg.get('package_classes',['backpack','handbag','suitcase','book'])); person_class=cfg.get('person_class','person')
    nearby_px=int(cfg.get('nearby_distance_px',180)); presence_min=int(cfg.get('presence_min_frames',10)); disappear_s=float(cfg.get('disappear_window_s',5.0))
    use_roi=bool(cfg.get('use_roi_presence',True)); roi_poly=cfg.get('package_roi',None); roi_min=float(cfg.get('roi_presence_min',12.0))
    last_pkg, last_person = deque(maxlen=int(cfg.get('proximity_grace_last_n',6))), deque(maxlen=int(cfg.get('proximity_grace_last_n',6)))
    state={'package_present_frames':0,'person_near_last_ts':None,'alerted':False,'last_seen_package_ts':None,'package_was_stable':False}
    logs=[]; events=[]; frame_idx=0
    while True:
        ok,frame=cap.read(); if not ok: break; t0=time.time()
        yres=model.predict(source=frame, conf=conf, iou=iou, device=device, verbose=False)[0]
        dets=[{'cls_name':yres.names[int(b.cls.item())],'conf':float(b.conf.item()),'bbox':b.xyxy[0].tolist()} for b in yres.boxes]
        package_boxes=[d['bbox'] for d in dets if d['cls_name'] in pkg_classes]; people_boxes=[d['bbox'] for d in dets if d['cls_name']==person_class]
        pkg_yolo=len(package_boxes)>0; gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); roi_present=False; roi_score=None
        if use_roi and roi_poly: roi_score=roi_presence_score(gray, roi_poly); roi_present = roi_score>=roi_min
        package_present = pkg_yolo or (roi_present if (use_roi and roi_poly) else False); person_present=len(people_boxes)>0
        if package_present: state['package_present_frames']+=1; state['last_seen_package_ts']=t0
        else: state['package_present_frames']=0
        pkg_stable = state['package_present_frames']>=presence_min
        if pkg_stable: state['package_was_stable']=True
        if package_boxes: [last_pkg.append(bbox_center_xyxy(b)) for b in package_boxes]
        if people_boxes: [last_person.append(bbox_center_xyxy(b)) for b in people_boxes]
        pkg_c=[bbox_center_xyxy(b) for b in package_boxes] if package_boxes else list(last_pkg)
        per_c=[bbox_center_xyxy(b) for b in people_boxes] if people_boxes else list(last_person)
        person_near=False; min_d=None
        if pkg_c and per_c:
            for pc in per_c:
                for kc in pkg_c:
                    d=l2(pc,kc); 
                    if (min_d is None) or (d<min_d): min_d=d
            person_near = (min_d is not None) and (min_d<=nearby_px)
        if person_near: state['person_near_last_ts']=t0
        alert=False
        if (state['person_near_last_ts'] is not None) and state['package_was_stable']:
            if (not package_present) and (state['last_seen_package_ts'] is not None):
                if (t0 - state['last_seen_package_ts'] <= disappear_s) and (t0 - state['person_near_last_ts'] <= disappear_s):
                    if not state['alerted']: alert=True; state['alerted']=True
        for d in dets:
            x1,y1,x2,y2=map(int,d['bbox']); color=(0,255,0) if d['cls_name']==person_class else (0,200,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{d['cls_name']} {d['conf']:.2f}",(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
        if use_roi and roi_poly:
            poly=np.array(roi_poly,np.int32); cv2.polylines(frame,[poly],True,(255,255,0),2)
            if roi_score is not None:
                x,y=poly[0]; cv2.putText(frame,f"roi:{roi_score:.1f}",(int(x),int(y)-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
        if min_d is not None: cv2.putText(frame,f"d={min_d:.1f}",(10,55),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        if person_near: cv2.putText(frame,"NEAR",(10,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame,f"pkg_stable:{pkg_stable} person_near:{person_near} alert:{alert}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        writer.write(frame)
        logs.append({'frame_idx':frame_idx,'timestamp_s':cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0,'package_present_yolo':pkg_yolo,'package_present_roi':bool(roi_present) if (use_roi and roi_poly) else False,'package_present':package_present,'package_stable':pkg_stable,'person_present':person_present,'person_near':person_near,'min_dist_px':min_d if min_d is not None else None,'alert':alert})
        if alert: events.append({'event':'POSSIBLE_THEFT','frame_idx':frame_idx,'timestamp_s':cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0})
        frame_idx+=1
    cap.release(); writer.release(); Path('outputs').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_csv('outputs/frame_signals.csv', index=False)
    pd.DataFrame(events).to_csv('outputs/alerts.csv', index=False)
    print(f'Saved: {out_path}; logs in outputs/')
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--video',required=True); ap.add_argument('--out',default='outputs/out_roi.mp4'); ap.add_argument('--config',default='configs/mvp.yaml'); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config,'r')); Path('outputs').mkdir(parents=True, exist_ok=True); run_pipeline(args.video, args.out, cfg)
