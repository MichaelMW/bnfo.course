#!/usr/bin/env python
# encoding: utf-8

## input matrix of samples * features. 
## output cluster
## method: 
##	representation : tsne OR pca
## 	label: from input OR kmean

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, decomposition
from sklearn.cluster import KMeans
import argparse


#### parse input
parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='inFile', help='input tsv file with Xs (featuers) and an optional Y (label); needs a header; needs first column as rownames')
parser.add_argument('-l', dest='label', help='columnName in the header used as the label column. If no label is found, will try to use cluster method to label the dots. If both -l and -m are provided, will remove -l column and use -m for labeling.')
parser.add_argument('-m', dest='mode', default="pca", help='mode of representation; pca (Default)  or tsne. If both -l and -m are provided, will remove -l column and use -m for labeling')
parser.add_argument('-p', dest='perplexity', default=30, help='perplexity for tsne, higher value leads to more evenly spaced points. Default: 30.')
parser.add_argument('-k', dest='n_clusters', default=False, help='number of clusters for kmeans; Default:False, using label for label')
parser.add_argument('-t', dest='outCluster', default='outCluster.tsv', help='output clusters; only works with -k. Default:outCluster.tsv')
parser.add_argument('-o', dest='outPlot', default = 'outPlot.pdf', help='output plot file; Default:outPlot.pdf')
parser.add_argument('-d', dest='hide_legend', action='store_true', default=False, help='hide legend; Default:False')
parser.add_argument('-a', dest='annotation', action='store_true', default=False, help='add annotation; Default:False')
parser.add_argument('-T', dest='transpose', action='store_true', default=False, help='transpose the data from using samples on the columns (and features/label on the rows) to samples on the rows(and features/label on the columns), Default=False')
args = parser.parse_args()

#### check args
inFile = args.inFile
label = args.label
mode = args.mode
perplexity = int(args.perplexity)
n_clusters = int(args.n_clusters) if args.n_clusters else False  # int or False
outCluster = args.outCluster
outPlot = args.outPlot
transpose = args.transpose
hide_legend = args.hide_legend
annotation = args.annotation

#### debug input parameters ####
#inFile = "input3.tsv"
#label = "survival"
#mode = "pca" # tsne
#n_clusters = 2
# perlexity = 30
# n_clusters = 2
# outFile = "cluster.pdf"

#### parameters ####
sep = "\t" # used to parse the inFile
header = 0 # use the first line as header
index_col = 0 # use the first column as the rowname
random_state = 0 # random state
n_components = 2 # 2D output

#### read input file
df = pd.read_csv(inFile, header = header, sep = sep, index_col = index_col)
if transpose:
	df = df.transpose()
fts = list(df.columns)
if label:
	y = df[label]
	fts.remove(label)
X = df[fts]
rnames = df.index.values

#### representation mode
## tsne
if mode == "tsne":
	tsne = manifold.TSNE(n_components = n_components, init = 'random', random_state = random_state, perplexity = perplexity)
	y_rep = tsne.fit_transform(X)

## pca
else:
	pca = decomposition.PCA(n_components = n_components)
	y_rep = pca.fit_transform(X)

## kmeans
if n_clusters:
	kw = "cluster: "
	y = KMeans(n_clusters = n_clusters, random_state = random_state).fit_predict(X)
	# write out kmeans clusters
	towrite = "\n".join([str(pair[0])+"\t"+str(pair[1]) for pair in zip(rnames, y)])
	with open(outCluster, "w") as f:
		f.write(towrite)
else:
	kw = "label: "

## plot
# main
y_sets = set(y)
for y_set in y_sets:
	indice = [index for index, eachY in enumerate(y) if eachY==y_set]
	px, py = zip(*y_rep[indice])
	plt.scatter(px, py, label = kw + str(y_set))

# legend
if not hide_legend:
	plt.legend()

# annotation
if annotation:
	px, py = zip(*y_rep)
	for i, rname in enumerate(rnames):
		plt.annotate(rname, (px[i], py[i]))

# savefig	
plt.savefig(outPlot)
