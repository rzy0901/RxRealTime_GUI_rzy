import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import math
import threading
from tkinter import *
from tkinter import font
from tkinter.ttk import Style
from ttkbootstrap import Style
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg)

def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result
    return func_wrapper

def PMR_fft(data_ref, data_tar):
    global fs, CIT, N_slide, T_slide
    fs = 1e6
    CIT = 0.1
    N_slide = 10
    T_slide = CIT / N_slide
    num_sample = round(fs*CIT)
    total_duration = round(len(data_ref) / fs)
    print(total_duration)
    array_start_time =np.arange(0,total_duration-CIT,T_slide)
    Doppler_frequency = 600
    step_dop = 1/CIT
    array_Doppler_frequency = np.arange(-Doppler_frequency-1,Doppler_frequency,step_dop)

    num_loop = len(array_Doppler_frequency)
    A_TD = np.zeros((len(array_start_time),num_loop),dtype = 'complex_')
    for idx in tqdm(range(len(array_start_time))):
    # for idx in range(len(array_start_time)):
        temp_tar = data_tar[round(array_start_time[idx]*fs):round(array_start_time[idx]*fs) + num_sample]
        temp_ref = data_ref[round(array_start_time[idx]*fs):round(array_start_time[idx]*fs) + num_sample]
        A_TD[idx,:] =  np.fft.fftshift(np.fft.fft(temp_ref*np.conj(temp_tar),num_sample))[round(num_sample/2-Doppler_frequency/step_dop-1):round(num_sample/2+Doppler_frequency/step_dop)]
        
    #这个地方是有问题的，日后记得改回来
    A_TD = np.abs(A_TD)
    return A_TD,array_Doppler_frequency,array_start_time

def show_real(array_start_time,array_Doppler_frequency,A_TD):
    pause_time = 0.2
    x, y = np.meshgrid(array_start_time,array_Doppler_frequency)
    plot_A_TD = np.zeros(np.size(A_TD))
    plt.figure(1)
    plt.ion()
    plot_A_TD = 20*(np.log(abs(A_TD)/np.max((np.max(abs(A_TD)))))/np.log(20))
    plt.clf()
    plt.pcolor(x,y,plot_A_TD.T,shading='auto',cmap='jet',vmin=-15,vmax=-5)        
    plt.ylabel('Doppler frequency (Hz)')
    plt.xlabel('time (s)')
    plt.pause(pause_time)
    plt.ioff()

def init(data):
    print(data.shape,type(data))
    data_sample = data
    # data_complex = data[0,:] + 1j*data[1,:]
    # data_sample = data.reshape(-1,2,order = "F")
    # data_sample = np.transpose(data_sample)
    #第一行是tar，第二行是ref
    print('[info] calibration\n')
    N = 10000
    data_tar = data_sample[1,:3*N]
    data_ref = data_sample[0,:3*N]
    h_corr = np.correlate(data_ref[:N*2+1],data_tar[:N *2+1],'same')
    h_corr = abs(h_corr)
    h_corr_max = np.argmax(h_corr) 
    array_sample_shift = h_corr_max - N 
    print("shift: ",array_sample_shift)
    if  array_sample_shift > 0:
        data_ref_cor = data_sample[0,array_sample_shift:]
        data_tar_cor = data_sample[1,:-array_sample_shift]
    else:
        data_ref_cor = data_sample[0,:array_sample_shift]
        data_tar_cor = data_sample[1,-array_sample_shift:]
    del data_ref,data_tar,data_sample
    return data_ref_cor,data_tar_cor

# @timer
def PMR_fft_show(data,A_TD_total,array_start_time_total):    
    global fs, CIT, N_slide, T_slide
    fs = 10e6
    CIT = 0.1
    N_slide = 2
    T_slide = CIT / N_slide
    data_ref, data_tar = init(data)
    A_TD,array_Doppler_frequency,array_start_time = PMR_fft(data_ref, data_tar)
    if not A_TD_total:
        print('111')
        A_TD_total = A_TD[:]
        array_start_time_total = array_start_time
    else:   
        A_TD_total = np.vstack((A_TD_total, A_TD))
        temp_time = array_start_time + array_start_time_total[-1] + T_slide
        array_start_time_total = np.hstack((array_start_time_total, temp_time))
    # print(A_TD_total.shape,array_start_time_total.shape)
    show_real(array_start_time_total,array_Doppler_frequency,A_TD_total)
    return A_TD_total,array_start_time_total
    # plt.show()
    
def PMR_fft3(data):
    data_ref, data_tar = init(data)
    A_TD,array_Doppler_frequency,array_start_time = PMR_fft(data_ref, data_tar)
    return A_TD,array_Doppler_frequency,array_start_time

def show_real_v2(array_start_time,array_Doppler_frequency,A_TD,Pred):
    pause_time = 5
    x, y = np.meshgrid(array_start_time,array_Doppler_frequency)
    plot_A_TD = np.zeros(np.size(A_TD))
    # plt.rcParams['font.sans-serif']=['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15,10))
    plt.ion()
    plot_A_TD = 20*(np.log(abs(A_TD)/np.max((np.max(abs(A_TD)))))/np.log(20))
    plt.clf()
    plt.pcolor(x,y,plot_A_TD.T,shading='auto',cmap='jet',vmin=-20,vmax=-10)      
    plt.colorbar()  
    plt.ylabel('Doppler frequency (Hz)',fontsize = 'yy-large')
    plt.xlabel('time (s)',fontsize='xx-large')
    if Pred  == 1:
        plt.title('Push Hand',fontsize='xx-large' )
    elif Pred == 2:
        plt.title('Fist',fontsize='xx-large')
    else:
        plt.title('Pinch Finger',fontsize='xx-large')
    plt.pause(pause_time)
    plt.ioff()

def gui(array_start_time,array_Doppler_frequency,plot_A_TD,Pred):
    a.clear()
    x, y = np.meshgrid(array_start_time,array_Doppler_frequency)
    a.pcolor(x,y,plot_A_TD.T,shading='auto',cmap='jet',vmin=-20,vmax=-5)      
    # a.set_colorbar()  
    a.set_ylabel('Doppler frequency (Hz)')
    a.set_xlabel('time (s)')
    if Pred  == 1:
        a.set_title('Push Hand',fontsize='xx-large' )
    elif Pred == 2:
        a.set_title('Fist',fontsize='xx-large')
    else:
        a.set_title('Pinch Finger',fontsize='xx-large')
    canvas.draw()

# def main_loop_gui(func):
#     global a,canvas,f
#     style = Style(theme='superhero')
#     # 注册窗口
#     win = style.master
#     win.title('Passive Sensing')
#     win.geometry('600x400')
#     win.resizable(0, 0)
#     f = Figure(figsize=(5, 4), dpi=100)
#     a = f.add_subplot(111)  # 添加子图:1行1列第1个
#     ft = font.Font(size=16, weight=NORMAL)
#     Start = Button(master=win,text='Start', command=func, font=ft)
#     Start.pack(side='top')
#     canvas = FigureCanvasTkAgg(f, master=win)  # A tk.DrawingArea.
#     canvas.get_tk_widget().pack(side=BOTTOM, fill='both', expand=1)
#     win.mainloop()

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