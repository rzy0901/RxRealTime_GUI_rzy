import numpy as np
import PMR_fft_v2 as Pf2
from GesturePred import ModelInit,GesPre
import matplotlib.pyplot as plt

def transforms(data):
    data = (data  - data.mean())/data.std()
    return data

def main():
    model = ModelInit()
    # fname = '/home/yuc/Rx_Ges/Dataset/Ges3_6.dat'
    fname = '/media/rzy/76800D98800D5FCB/Codes/RxRealTime_GUI_rzy/data1.dat'
    data = np.fromfile(fname,dtype = np.complex64)
    data_sample = data.reshape(-1,2,order = "F")
    data_sample = np.transpose(data_sample)
    A_TD,array_Doppler_frequency,array_start_time =Pf2.PMR_fft3(data_sample)
    plot_A_TD = 20*(np.log(abs(A_TD)/np.max((np.max(abs(A_TD)))))/np.log(20))
    # plot_A_TD = transforms(plot_A_TD)
    # pause_time = 3
    # total_duration = 2
    # CIT = 0.1
    # T_slide = 0.01
    # array_start_time =np.arange(0,total_duration-CIT,T_slide)
    # Doppler_frequency = 600
    # step_dop = 1/CIT
    # array_Doppler_frequency = np.arange(-Doppler_frequency-1,Doppler_frequency,step_dop)
    # x, y = np.meshgrid(array_start_time,array_Doppler_frequency)
    # plt.figure(1)
    # plt.ion()
    # plt.clf()
    # plt.pcolor(x,y,plot_A_TD.T,shading='auto',cmap='jet')        
    # plt.colorbar()
    # plt.ylabel('Doppler frequency (Hz)')
    # plt.xlabel('time (s)')
    # plt.pause(pause_time)
    # plt.ioff()
    # plot_A_TD[plot_A_TD<-15] = -15
    # plot_A_TD[plot_A_TD>-5] = -5
    Pred = GesPre(plot_A_TD.T,model)
    print(Pred)


    Pf2.show_real_v2(array_start_time,array_Doppler_frequency,A_TD,Pred)

if __name__ == '__main__':
    main()