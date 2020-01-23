#!/bin/sh

#### examples 


### supervised learning
./gb.py -i input1.tsv -l price -m regression # simple example, play with input
./gb.py -i input2.tsv -l survival -m regression # regresssion
./gb.py -i input3.tsv -l survival -m classification   # classification, real data
./gb.py -i input4.tsv -l survival -m classification   # classification, randome data as control

### unsupervised learning
## using input survival as label
./cluster.py -i input3.tsv -l survival -o l.pdf
## using kmean clustering results as label
# kmean #cluster = 3
./cluster.py -i input3.tsv -l survival -k 3 -o k3.pdf
# kmean #cluster = 4
./cluster.py -i input3.tsv -l survival -k 4 -o k4.pdf
# use tsne
./cluster.py -i input3.tsv -l survival -m tsne -k 4 -o k4.tsne.pdf
# kmean #cluster = 4, with annotation
./cluster.py -i input3.tsv -l survival -k 4 -o k4.anno.pdf -a

