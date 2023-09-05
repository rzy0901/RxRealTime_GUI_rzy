import sys
import time
import threading
import logging
import numpy as np
import uhd
import datetime
import socket
import subprocess
import matplotlib.pyplot as pl

# import gui 

# from PMR_fft import PMR_fft
import PMR_fft_v2 as Pf2
from GesturePred import ModelInit,GesPre


INIT_DELAY = 0.1  # 50mS initial delay before transmit
recv_buffer = np.zeros((2,2500), dtype=np.complex64)
file_buffer = np.zeros((2,2500), dtype=np.complex64)


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Command line options.')
    parser.add_argument("-a", "--args", default="addr=192.168.10.2", type=str, help="single uhd device address args")
    parser.add_argument("--rate", type=float, default=1e6,help="IQ rate(sps)") 
    parser.add_argument("--gain", type=float, default=0,help="gain") 
    return parser.parse_args()

def uhd_builder(args="addr=192.168.10.2",gain=0.0,rate=1e6):
    usrp=uhd.usrp.MultiUSRP(args)
    usrp.set_clock_source("internal")
    usrp.set_time_source("internal")
    usrp.set_time_now(uhd.types.TimeSpec(0.0))

    usrp.set_rx_rate(rate,0)
    usrp.set_rx_rate(rate,1)
    usrp.set_rx_gain(gain,0)
    usrp.set_rx_gain(gain,1)
    usrp.set_rx_antenna("RX2",0)
    usrp.set_rx_antenna("RX2",1)
    usrp.set_rx_bandwidth(20e6,0)
    usrp.set_rx_bandwidth(20e6,1)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(500e6), 0)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(500e6), 1)
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels=[0,1]
    
    return[usrp,st_args]

def AE_init():
    socket_tx=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # addr_port_tx=("192.168.43.20", 5221)
    addr_port_tx=("192.168.43.85", 5220)
    socket_rx_ref = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr_port_rx_ref = ("127.0.0.1", 5222)
    socket_rx_sur = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr_port_rx_sur = ("127.0.0.1", 5220)
    print('[info] socket init is OK.')

###########################
    tx_gain = 15
    rx_ref_gain = 12
    rx_tar_gain = 15
#############################

    gain_tx_bf = tx_gain
    gain_tx_rf = tx_gain
    gain_rx_ref_bf =rx_ref_gain
    gain_rx_ref_rf =rx_ref_gain
    gain_rx_sur_bf = rx_tar_gain
    gain_rx_sur_rf = rx_tar_gain

    filename_beambook_tx = '-1'
    filename_beambook_rx_ref = '-1'
    filename_beambook_rx_sur = '-1'
    idx_pattern_tx = 32  # 32
    idx_pattern_rx_ref = 32 # 32
    idx_pattern_rx_sur = 32 # 32
    if_AE_on_tx = -1 # -1
    if_AE_on_rx_ref = -1 # -1
    if_AE_on_rx_sur = -1 # -1

    packet_tx = '{0},{1},{2},{3},{4}'.format(gain_tx_bf,gain_tx_rf,filename_beambook_tx,idx_pattern_tx,if_AE_on_tx)
    socket_tx.sendto(packet_tx.encode(),addr_port_tx)
    time.sleep(0.2)
    packet_rx_ref = '{0},{1},{2},{3},{4}'.format(gain_rx_ref_bf,gain_rx_ref_rf,filename_beambook_rx_ref,idx_pattern_rx_ref,if_AE_on_rx_ref)
    socket_rx_ref.sendto(packet_rx_ref.encode(),addr_port_rx_ref)
    time.sleep(0.2)
    packet_rx_sur = '{0},{1},{2},{3},{4}'.format(gain_rx_sur_bf,gain_rx_sur_rf,filename_beambook_rx_sur,idx_pattern_rx_sur,if_AE_on_rx_sur)
    socket_rx_sur.sendto(packet_rx_sur.encode(),addr_port_rx_sur)
    time.sleep(0.2)
    print('[info] Socket send complete.')
    socket_tx.close()
    socket_rx_ref.close()
    socket_rx_sur.close()

def rx_host_2chan(usrp,st_args):
    #### gpio ###################
    # BF_Res = 0x1  
    # #Reset
    # BF_Inc = 0x4  #LED
    # #BF_inc
    # GPIO = (BF_Res | BF_Inc)
    # all_one = 0xFF
    # all_zero = 0x00

    # #init
    # usrp.set_gpio_attr("FP0","DDR",all_one,GPIO,0)
    # usrp.set_gpio_attr("FP0","CTRL",all_zero,GPIO,0)
    #####
    channels = [0,1]
    ############################
    duration = 2
    ############################
    num_sample = int(np.ceil(duration*usrp.get_rx_rate()))
    result = np.empty((len(channels), num_sample), dtype=np.complex64)

    metadata = uhd.types.RXMetadata()
    streamer = usrp.get_rx_stream(st_args)
    buffer_samps = streamer.get_max_num_samps()
    recv_buffer = np.zeros((len(channels), buffer_samps), dtype=np.complex64)
    # print("[info] Start sampling in 1s.")
    # time.sleep(1)

    recv_samps = 0
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = False
    stream_cmd.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + 0.1)
    streamer.issue_stream_cmd(stream_cmd)
    samps = np.array([], dtype=np.complex64)

    # print(time.strftime('%Y/%m/%d %H:%M:%S',time.localtime(time.time())))
    # usrp.set_gpio_attr("FP0","OUT",BF_Inc,BF_Inc,0)
    
    while recv_samps < num_sample:
        samps = streamer.recv(recv_buffer, metadata)
        if samps:
            real_samps = min(num_sample - recv_samps, samps)
            result[:, recv_samps:recv_samps + real_samps] = recv_buffer[:, 0:real_samps]
            recv_samps += real_samps
    # print("[succ] USRP samping complete.\n")
    # usrp.set_gpio_attr("FP0","OUT",all_zero,BF_Inc,0)
    print(time.strftime('%Y/%m/%d %H:%M:%S',time.localtime(time.time())))
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(stream_cmd)
    streamer = None
    # filename = '/home/yuc/Rx_Ges/Dataset/Ges3_'+str(index)+'.dat'
    # with open(filename, 'wb') as fid:
    #     result.tofile(fid)
    # print("[succ] File saving complete.\n")
    return result

def main_loop_gui():
        data = rx_host_2chan(usrp,st_args)
        A_TD,array_Doppler_frequency,array_start_time =Pf2.PMR_fft3(data)
        plot_A_TD = 20*(np.log(abs(A_TD)/np.max((np.max(abs(A_TD)))))/np.log(20))
        Pred = GesPre(plot_A_TD.T,model)
        Pf2.gui(array_start_time,array_Doppler_frequency,plot_A_TD,Pred)

def main():
    global usrp,model,st_args
    args = get_args()
    subprocess.Popen(["bash", "start_v3_ref.sh","open"])
    time.sleep(10)
    subprocess.Popen(["bash", "start_v3_sur.sh","open"])
    time.sleep(10)

    AE_init()
    print('RF initialize success')

    model = ModelInit()
    print('Model initialize success')

    usrp,st_args = uhd_builder(args.args,args.gain,args.rate)
    print('USRP initialize success')
    Pf2.main_loop_gui(main_loop_gui)



if __name__ == "__main__":
    main()
