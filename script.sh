#!/bin/sh

./gb.py -i input1.tsv -l price -m regression
./gb.py -i input2.tsv -l survival -m regression
./gb.py -i input3.tsv -l survival -m classification
./gb.py -i input4.tsv -l survival -m classification

