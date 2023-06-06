#!/bin/bash

rm -rf ../dbs/*
for i in ` cat ../docs/list.txt`; 
do 
	echo "*** $i ***"
	python3 chat.py new ../docs/$i ../dbs/$i 
	cp ../docs/$i/* ../docs/all/
done
python3 chat.py new ../docs/all ../dbs/all

