import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from tkinter import *
from tkinter import font
from tkinter.ttk import Style
from ttkbootstrap import Style
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg)

def gui(Pred):
    a.clear()
    img = mpimg.imread('./temp_matlab.jpg')
    a.imshow(img)
    a.axis('off')    
    if Pred  == 0:
        a.set_title('Push & Pull',fontsize='xx-large' )
    elif Pred == 1:
        a.set_title('Beckoned',fontsize='xx-large')
    else:
        a.set_title('Rub Finger',fontsize='xx-large')
    canvas.draw()

def main_loop_gui(func):
    global a, canvas, f
    style = Style(theme='superhero')
    # 注册窗口
    win = style.master
    win.title('MmWave Communication and Passive Sensing System')
    w = win.winfo_screenwidth()
    h = win.winfo_screenheight()
    win.geometry('%sx%s'%(w,h))
    # win.resizable(0, 0)
    # 界面一  # 界面二
    f = Figure(figsize=(9,6), dpi=100)
    a = f.add_subplot(111)  # 添加子图:1行1列第1个
    ft = font.Font(size=18, weight=NORMAL)
    Start = Button(win, text='Start', command=func, font=ft).pack(side=BOTTOM, ipadx=100,ipady=30)
    canvas = FigureCanvasTkAgg(f, win)  # A tk.DrawingArea.
    width1=30
    label5 = Label(win, text="",
                   # 设置标签内容区大小
                   width=int(width1*0.2), height=5,font=ft).pack(side=RIGHT, fill=Y)
    canvas.get_tk_widget().pack(side=RIGHT, fill='both', expand=1)
    # 标签
    label1 = Label(win, text="Duration:  2 s",
                   # 设置标签内容区大小
                   width=width1, height=5,font=ft).pack(side=BOTTOM, fill=Y,expand=YES)
    label2 = Label(win, text="Sampling Rate: 1 M/s",
                   # 设置标签内容区大小
                   width=width1, height=5,font=ft).pack(side=BOTTOM, fill=Y,expand=YES)
    label3 = Label(win, text="Carrier Frequency: 60.48 GHz ",
                   # 设置标签内容区大小
                   width=width1, height=5,font=ft).pack(side=BOTTOM, fill=Y,expand=YES)
    label4 = Label(win, text="CIT : 0.1 s",
                   # 设置标签内容区大小
                   width=width1, height=5,font=ft).pack(side=BOTTOM, fill=Y,expand=NO)
    win.mainloop()