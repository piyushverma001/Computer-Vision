import cv2
import numpy as np


vid = cv2.VideoCapture(0)


while True:
    _,cap = vid.read()

    hsv_raw = cv2.cvtColor(cap,cv2.COLOR_BGR2HSV)
    hsv = cv2.medianBlur(hsv_raw,5)
    gray = cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
    
    max_red=np.array([10,256,256])
    min_red=np.array([0,150,80])


    mask1=cv2.inRange(hsv,min_red,max_red)

    max_red2=np.array([180,256,256])
    min_red2=np.array([170,150,80])


    mask2=cv2.inRange(hsv,min_red2,max_red2)
    

    mask =mask1+mask2


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    #morph the image. closing operation Dilation followed by Erosion. 
    #It is useful in closing small holes inside the foreground objects, 
    #or small black points on the object.
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #erosion followed by dilation. It is useful in removing noise
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

	#coloured = cv2.
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    if len(contours)>0:
        segmented = max(contours,key=cv2.contourArea)
        cv2.drawContours(cap, segmented, -1,0, -1)
        ellipse = cv2.fitEllipse(segmented)
        #add it
        cv2.ellipse(cap, ellipse, (0,255,0), 2, cv2.LINE_AA)
    
    cv2.imshow("HSV converted Feed",hsv)
    cv2.imshow("Masked Feed",mask_clean)
    cv2.imshow("Original Feed",cap)
    key = cv2.waitKey(1)


    if key==ord('q'):
    	break

vid.release()
cv2.destroyAllWindows()
