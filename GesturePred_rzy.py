import numpy as np
from PMR_fft import PMR_fft
import subprocess


def main():
    filename = './data1.dat'
    data = np.fromfile(filename,dtype = '<f', count = -1,).reshape(2,-1,order = "F")
    data_complex = data[0,:] + 1j*data[1,:]
    data_sample = data_complex.reshape(-1,2,order = "F")
    data_sample = np.transpose(data_sample)
    PMR_fft(data_sample,noAxes=True)
    # 或者terminal使用如下代码
    # matlab -nodisplay -nosplash -r "func_PMR_fft('./data1.dat','./temp_matlab.jpg'); exit
    matlab_command = "matlab -nodisplay -nosplash -r \"func_PMR_fft('{}','./temp_matlab.jpg'); exit;\"".format(filename)
    print(matlab_command)
    try:
        subprocess.run(matlab_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
if __name__ == '__main__':
    main()