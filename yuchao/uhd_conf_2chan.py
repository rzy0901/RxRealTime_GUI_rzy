import uhd
import logging

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
