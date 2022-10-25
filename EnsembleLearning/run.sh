#!/usr/bin/bash

# import ml module if not imported already
if [ ! -d "ml" ];then
  cp -r ../ml ./
fi

# generate ada boost python script if it doesn't exists
if [ ! -f "ada_boost.py" ];then
	jupyter nbconvert --to python ada_boost.ipynb
fi
python3 ada_boost.py

#generate bagging python script if it doesn't exists
if [ ! -f "bagging1.py" ];then
  jupyter nbconvert --to python bagging1.ipynb
fi
python3 bagging1.py

#generate random forest python script if it doesn't exists
if [ ! -f "random_forest.py" ];then
  jupyter nbconvert --to python random_forest.ipynb
fi
python3 random_forest.py
