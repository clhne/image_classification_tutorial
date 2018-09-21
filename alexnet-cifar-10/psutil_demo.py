# -*- coding: utf-8 -*-
import psutil
import os
import sys
import time

pid = os.getpid()

p = psutil.Process(pid)
print ('Process info:')
print ('name: ', p.name())
print ('exe:  ', p.exe())

data = []

while True:
    data += list(range(100001))
    info = p.memory_full_info()
    memory = info.uss / 1024. / 1024.
    print ('Memory used: {:.2f} MB'.format(memory))
    '''
    if memory > 40:
        print ('Memory too big! Exiting.')
        sys.exit()
    '''
    time.sleep(1)
