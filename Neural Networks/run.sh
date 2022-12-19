#!/usr/bin/bash

# import ml module if not imported already
if [ ! -d "ml" ];then
  cp -r ../ml ./
fi

# generate python script if it doesn't exists
echo "2a>"
if [ ! -f "2a.py" ];then
	jupyter nbconvert --to python 2a.ipynb
fi
python3 2a.py
echo "-----------------------------------------------------------------------------------------------------------------"
echo ""
echo ""

# generate python script if it doesn't exists
echo "2b>"
if [ ! -f "2b.py" ];then
	jupyter nbconvert --to python 2b.ipynb
fi
python3 2b.py
echo "-----------------------------------------------------------------------------------------------------------------"
echo ""
echo ""

# generate python script if it doesn't exists
echo "2c>"
if [ ! -f "2c.py" ];then
	jupyter nbconvert --to python 2c.ipynb
fi
python3 2c.py
echo "-----------------------------------------------------------------------------------------------------------------"
echo ""
echo ""

# generate python script if it doesn't exists
echo "bonus>"
if [ ! -f "bonus.py" ];then
	jupyter nbconvert --to python bonus.ipynb
fi
python3 bonus.py
