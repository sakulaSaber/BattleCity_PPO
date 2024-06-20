import cv2
from screen_test import grab_screen
window1 = (209,335,727,853)
#(10,91,640,685)
#(959,369,1055,385)
while(True):

    screen_gray = cv2.cvtColor(grab_screen(window1),cv2.COLOR_BGR2GRAY)#灰度图像收集
    
    cv2.imshow('window1',screen_gray)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

