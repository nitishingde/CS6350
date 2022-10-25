#!/usr/bin/bash

# import ml module if not imported already
if [ ! -d "ml" ];then
  cp -r ../ml ./
fi

# generate lms1 python script if it doesn't exists
if [ ! -f "lms1.py" ];then
	jupyter nbconvert --to python lms1.ipynb
fi
python3 lms1.py

#generate lms2 python script if it doesn't exists
if [ ! -f "lms2.py" ];then
  jupyter nbconvert --to python lms2.ipynb
fi
python3 lms2.py

#generate lms3 python script if it doesn't exists
if [ ! -f "lms3.py" ];then
  jupyter nbconvert --to python lms3.ipynb
fi
python3 lms3.py
