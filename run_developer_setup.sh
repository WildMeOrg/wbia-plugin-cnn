#!/bin/bash

./clean.sh
python setup.py clean

pip install --user -r requirements.txt

python setup.py develop

pip install --user -e .

pip install --user -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
pip install --user https://github.com/Lasagne/Lasagne/archive/master.zip

#cd Lasagne/
#pip install -e .

cd ../
