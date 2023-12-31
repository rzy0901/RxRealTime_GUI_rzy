#!/bin/bash

:<<EOF
function help { pushd .; cd Eder_B; python eder.py -h; popd; }

go_rundir="cd Eder_B"
python_cmd="python -i eder_v2.py"
other_opt=""
while [[ $# > 0 ]];
do
    case "$1" in
      -h|--help)        help; read -n 1 -s; exit;;
      -g|--gui)         go_rundir="cd Eder_B/pythonGUI"; python_cmd="python viewNotebook.py"; shift;; 
      -f=*|--fref=*)    other_opt=${other_opt}" -f ${1#*=}"; shift;;
      -f|--fref)        shift; other_opt=${other_opt}" -f ${1}"; shift;;
      -r=*|--rfm=*)     other_opt=${other_opt}" -r ${1#*=}"; shift;;
      -r|--rfm)         shift; other_opt=${other_opt}" -r ${1}"; shift;;
      -u=*|--unit=*)    other_opt=${other_opt}" -u ${1#*=}"; shift;;
      -u|--unit)        shift; other_opt=${other_opt}" -u ${1}"; shift;;
      -b=*|--board=*)   other_opt=${other_opt}" -b ${1#*=}"; shift;;
      -b|--board)       shift; other_opt=${other_opt}" -b ${1}"; shift;;
      *)                other_opt=${other_opt}" -u ${1}"; shift;;
  esac
done


{
sudo modprobe -r ftdi_sio
} &> /dev/null


export LD_LIBRARY_PATH=/usr/local/lib
$go_rundir
echo $python_cmd$other_opt
$python_cmd$other_opt
EOF

###
cd ../Eder_B
sudo python eder_v2.py -f 45e6 -b MB1 -u SN0369 -r BFM06005 -x RX -p 5222
