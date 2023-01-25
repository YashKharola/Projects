import cv2
import numpy as np
logo=cv2.imread("logo2.png")
h_logo,w_logo,_=logo.shape
img=cv2.imread("images.png")
h_img,w_img,_=img.shape

center_y=int(h_img/2)
center_x=int(w_img/2)
top_y=center_y -int(h_logo/2)
left_x=center_x -int(w_logo/2)
bottom_y=top_y+h_logo
right_x=left_x+w_logo

roi=img[top_y:bottom_y,left_x:right_x]

result=cv2.addWeighted(roi,1,logo,0.6,0)
img[top_y:bottom_y,left_x:right_x]=result
cv2.imshow("Logo",logo)
cv2.imshow("Img",img)
cv2.imshow("Roi",roi)
cv2.imshow("Result",result)
cv2.waitKey(0)
