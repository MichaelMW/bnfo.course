#!/bin/sh

./gb.py -i input1.tsv -l price -m regression
./gb.py -i input2.tsv -l survival -m regression

