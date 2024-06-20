import keyboard
import time

def attack():
    keyboard.press('j')
    time.sleep(0.1)
    keyboard.release('j')

def go_forward():
    keyboard.press('w')
    time.sleep(0.1)
    keyboard.release('w')

def go_back():
    keyboard.press('s')
    time.sleep(0.1)
    keyboard.release('s')

def go_left():
    keyboard.press('a')
    time.sleep(0.1)
    keyboard.release('a')

def go_right():
    keyboard.press('d')
    time.sleep(0.1)
    keyboard.release('d')

def go_forward_and_attack():
    keyboard.press('w')
    keyboard.press('j')
    time.sleep(0.1)
    keyboard.release('w')
    keyboard.release('j')
    
def go_back_and_attack():
    keyboard.press('s')
    keyboard.press('j')
    time.sleep(0.1)
    keyboard.release('s')
    keyboard.release('j')
    
def go_left_and_attack():
    keyboard.press('a')
    keyboard.press('j')
    time.sleep(0.1)
    keyboard.release('a')
    keyboard.release('j')
    
def go_right_and_attack():
    keyboard.press('d')
    keyboard.press('j')
    time.sleep(0.1)
    keyboard.release('d')
    keyboard.release('j')
    
#暂停用的，其实没什么卵用
def press_e():
    keyboard.press_and_release('e')