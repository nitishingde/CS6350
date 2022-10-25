#!/usr/bin/bash

# import ml module if not imported already
if [ ! -d "ml" ];then
  cp -r ../ml ./
fi

# generate cars python script if it doesn't exists
if [ ! -f "cars.py" ];then
	jupyter nbconvert --to python cars.ipynb
fi
python3 cars.py

#generate bank python scrip if it doesn't exists
if [ ! -f "bank.py" ];then
  jupyter nbconvert --to python bank.ipynb
fi
python3 bank.py
