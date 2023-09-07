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

import GUI as gui
from GesturePred_rzy import ModelInit,GesPre,PMR_fft_matlabEngine,testTransform



def main():
    global model,eng
    model = ModelInit()
    import matlab.engine
    eng = matlab.engine.start_matlab()
    def main_loop_gui():
        # data = rx_host_2chan(usrp,st_args)
        data = np.load('data1.npy')
        img = PMR_fft_matlabEngine(eng,data,imshow=False)
        img_transfrom = testTransform(img)
        Pred = GesPre(img_transfrom,model)
        gui.gui(int(Pred))
    print('Model initialize success')
    gui.main_loop_gui(main_loop_gui)
    print("Matlab engine start")
    eng.quit()

if __name__ == "__main__":
    main()
