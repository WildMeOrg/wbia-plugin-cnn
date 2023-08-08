#!/bin/bash

./clean.sh
python setup.py clean

pip install -r requirements.txt

python setup.py develop

pip install -e .

pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
pip install https://github.com/Lasagne/Lasagne/archive/master.zip

#cd Lasagne/
#pip install -e .

cd ../
