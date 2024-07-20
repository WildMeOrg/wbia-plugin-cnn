#!/bin/bash

./clean.sh
python3 setup.py clean

pip3 install -r requirements.txt

python3 setup.py develop

pip3 install -e .

pip3 install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
pip3 install https://github.com/Lasagne/Lasagne/archive/master.zip

#cd Lasagne/
#pip install -e .

cd ../
