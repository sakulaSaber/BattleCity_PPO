# -*- coding: utf-8 -*-

import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

# 添加ESC键的虚拟键码
keyList.append(27)

def key_check():
    keys = []
    for key in keyList:
        # 如果key是integer类型（即虚拟键码）
        if isinstance(key, int):  
            if wapi.GetAsyncKeyState(key):
                keys.append('ESC')  # 添加对应的字符
        # 如果key是普通字符
        else:
            if wapi.GetAsyncKeyState(ord(key)):
                keys.append(key)

    return keys
