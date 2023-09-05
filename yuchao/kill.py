'''
Description: 
Author: Yu Chao 
Date: 2022-03-15 16:05:11
LastEditTime: 2022-07-04 13:08:32
'''
import os



def killport(port):
    command = 'sudo lsof -i:' + str(port)
    password = '1112'
    M = os.popen('echo %s | sudo -S %s' % (password,command)).readlines()
    PID = M[1].split(' ')
    for i in PID:
        if i == '' or i == ' ':
            PID.remove(i)
    kill_cmd = 'sudo kill -9 ' + PID[1]
    os.system(kill_cmd)
    print('The port of %s was killed',port)

def main():
    ref_port = 5220
    tar_port = 5222
    try:
        killport(ref_port)
    except:
        pass
    killport(tar_port)


if __name__ == '__main__':
    main()
    
    
