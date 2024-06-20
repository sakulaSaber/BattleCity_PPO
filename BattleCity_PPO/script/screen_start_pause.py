import time 
from pynput.keyboard import Controller, Key

def start_game():

    # 首先暂停1秒
    time.sleep(1)

    # 创建一个键盘控制器
    keyboard = Controller()

    keyboard.press('1')
    time.sleep(0.5)
    keyboard.release('1')
        
    print("游戏开始")
    
#停止游戏
def is_pause():
    # 创建一个键盘控制器
    keyboard = Controller()
    keyboard.press(Key.esc)
    keyboard.release(Key.esc)
    
#start_game()