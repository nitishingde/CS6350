#!/usr/bin/bash

# import ml module if not imported already
if [ ! -d "ml" ];then
  cp -r ../ml ./
fi

# generate python script if it doesn't exists
if [ ! -f "2.py" ];then
	jupyter nbconvert --to python 2.ipynb
fi
python3 2.py
