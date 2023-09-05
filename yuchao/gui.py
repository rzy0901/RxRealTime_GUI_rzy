from tkinter import *
from tkinter import font
from tkinter import Label
from tkinter.ttk import Style
from ttkbootstrap import Style
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg)
import time


# def init():
#     # global win,head_button,ft
#     style = Style(theme='darkly')
#     # 注册窗口
#     win = style.master
#     win.title('Passive Sensing')
#     win.geometry('600x400')
#     win.resizable(0, 0)
#     # 字体
#     return win


# 点了计算调用函数！！作为一个接口
def calculate():
    a.clear()
    matrix = np.random.random((100, 200))
    x = np.arange(0, 3, 0.01) * np.random.randint(1, 300)
    y = np.sin(2 * np.pi * x)
    # 在前面得到的子图上绘图
    a.pcolor(matrix)
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
    ft = font.Font(size=30, weight=NORMAL)
    Start = Button(win, text='Start', command=func, font=ft).pack(side=BOTTOM, ipadx=50)
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


# while 1:
# 计算按钮
# def main():
# init()
# global a,canvas,f
# f = Figure(figsize=(5, 4), dpi=100)
# a = f.add_subplot(111)  # 添加子图:1行1列第1个

# Start = Button(master=win,text='Start', command=calculate, font=ft)
# Start.pack(side='top')
# canvas = FigureCanvasTkAgg(f, master=win)  # A tk.DrawingArea.
# canvas.get_tk_widget().pack(side=BOTTOM, fill='both', expand=1)
# win.mainloop()


# # 
# if __name__  == 'main':
main_loop_gui(calculate)
