import numpy as np
from script.screen_test import grab_screen



def mytank():
    window1 = (209,335,727,853)
    screen = grab_screen(window1)
    
    #screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    # 寻找像素块为 rgb 的像素（只比较 RGB 通道，忽略透明度通道）；我方坦克像素rgb(255, 162, 67)
    # 这里是bgr顺序，所以要反过来(67, 162, 255)
    mask = np.all(screen[:,:,:3] == (67, 162, 255), axis=-1)
    pixels_found = np.sum(mask)
    # print(pixels_found)
    # print(pixels_found)

    if pixels_found != 0: #我方存在
        return True 
    else: #我方不存在
        return False 
    
""" while True:
    if mytank():
        print("存在")
    else:
        print("不存在") """