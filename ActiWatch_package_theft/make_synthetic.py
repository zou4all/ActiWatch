import cv2, numpy as np, os
os.makedirs('data',exist_ok=True)
w,h=1280,720
out=cv2.VideoWriter('data/your_clip.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (w,h))
pkg=(900,560,1000,640)
for i in range(250):
    f=np.zeros((h,w,3),np.uint8)+40
    if i<160: cv2.rectangle(f,(pkg[0],pkg[1]),(pkg[2],pkg[3]),(0,150,200),-1)
    x=int(200+i*3.2); x=max(60,min(w-60,x)); y1,y2=420,690
    cv2.rectangle(f,(x,y1),(x+60,y2),(0,200,100),-1)
    if abs(x-pkg[0])<80: cv2.circle(f,(pkg[0]+10,pkg[1]+20),10,(0,0,255),-1)
    cv2.putText(f,f'frame {i}',(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)
    out.write(f)
out.release(); print('Wrote data/your_clip.mp4')
