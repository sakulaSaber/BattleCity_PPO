from cnocr import CnOcr
import cv2
from script.screen_test import grab_screen

window1 = (959,369,1055,385) #数字识别窗口

def num_rec(window):

    # 创建一个OCR对象
    ocr = CnOcr(det_model_name='model\cnocr-v2.3-number-densenet_lite_136-fc-epoch=023.ckpt')

    #while(True):
        
    # 截取图像
    screen = grab_screen(window)
    
    # 如果图像是四通道的，将其转换为三通道的BGR图像
    if screen.shape[2] == 4:
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

    # 利用OCR进行数字识别
    ocr_result = ocr.ocr(screen)
    
    
    # 打印识别结果
    if ocr_result:
        if ocr_result[0]['text'].isdigit():
            #print(ocr_result[0]['text'])
            return int(ocr_result[0]['text'])
        else:
            #print('The value of ocr_result[0][\'text\'] is not a number.')
            return -1 # 或者返回一个默认值
    else:
        #print('No OCR result found.')
        return -1 # 或者返回一个默认值
    
    """ if ocr_result:
        print(ocr_result[0]['text'])
        return int(ocr_result[0]['text'])
    else:
        print("NULL")
        return -1
     """

     
    """ cv2.imshow('window1', screen)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break """
