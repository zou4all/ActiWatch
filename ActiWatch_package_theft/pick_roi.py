import cv2, sys, json
video=sys.argv[1] if len(sys.argv)>1 else 'data/your_clip.mp4'
cap=cv2.VideoCapture(video)
ok,frame=cap.read(); cap.release()
assert ok,'Cannot read first frame'
pts=[]

def cb(e,x,y,f,p):
    if e==cv2.EVENT_LBUTTONDOWN:
        pts.append([int(x),int(y)]); print('Clicked:',[x,y]); cv2.circle(frame,(x,y),5,(0,255,0),-1); cv2.imshow('click 4 points',frame)
cv2.imshow('click 4 points',frame); cv2.setMouseCallback('click 4 points',cb)
print('Click 4 corners; press q to finish')
while True:
    k=cv2.waitKey(20)&0xFF
    if k==ord('q') or len(pts)>=4: break
cv2.destroyAllWindows(); print('\nPaste into YAML:'); print('  - ' + json.dumps(pts))
