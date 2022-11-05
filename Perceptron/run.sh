#!/usr/bin/bash

# import ml module if not imported already
if [ ! -d "ml" ];then
  cp -r ../ml ./
fi

# generate bank python script if it doesn't exists
if [ ! -f "bank.py" ];then
	jupyter nbconvert --to python bank.ipynb
fi
python3 bank.py

