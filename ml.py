#!/usr/bin/env python
# encoding: utf-8

### todo
### import more models. 


### use gradient boosting. 
### notice, importance always come from full model using all data. (CV is irrelevant here) 

import numpy as np

## import args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='inFile', help='input tsv file with Xs (featuers) and Y (label); needs a header; needs first column as rownames')
parser.add_argument('-l', dest='label', help='columnName in the header used as the label column.')
parser.add_argument('-m', dest='mode', default="classification", help='regression or classification.')
parser.add_argument('-n', dest='cvFold', default=0, help='number of CV fold; use 0 if no CV is needed. Default:0')
parser.add_argument('-t', dest='ntree', default=100 , help='number of trees. Default:100')
parser.add_argument('-p', dest='ncpu', default=-1 , help='number of CPU process. Default:1')
parser.add_argument('-I', dest='impFile', default="impFile.tsv" , help='output importance file. Default: impFile.tsv')
parser.add_argument('-P', dest='predFile', default="predFile.tsv" , help='output Y-pred file. Default: predFile.tsv')
parser.add_argument('-L', dest='lasso', default=False , help='use lasso, default to False; otherwise use a number for alpha: 0 to 1; 0 for an ordinary least square, 1 for conanical L1 lasso. Default: False')
args = parser.parse_args()

## check args
inFile = args.inFile
label = args.label
ntree = int(args.ntree)
ncpu = int(args.ncpu)
cvFold = int(args.cvFold)
mode = args.mode
impFile = args.impFile
predFile = args.predFile
lasso = float(args.lasso) if args.lasso else False

## default values
sep = "\t" # used to parse the inFile
header = 0 # use the first line as header
index_col = 0 # use the first column as the rowname
random_state = 0 # random state

## debug
#inFile = "data.more.tsv"
#label = "TMP"
#ntree = 100
#ncpu = -1
#cvFold = 5
#mode = "regression"
#impFile = "testImp.tsv"

## read inFile
import pandas as pd
df = pd.read_csv(inFile, header = header, sep = sep, index_col = index_col)
y = df[label]
fts = list(df.columns)
fts.remove(label)
X = df[fts]

## ML model
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
if mode == "classification":
	model = GradientBoostingClassifier(n_estimators = ntree, random_state=random_state, max_features="log2", max_depth=3)
else:
	model = GradientBoostingRegressor(n_estimators = ntree, random_state=random_state, max_features="log2", max_depth=3)

## performance eval
from numpy import mean, std
from sklearn.metrics import roc_auc_score, average_precision_score, explained_variance_score, r2_score
from scipy.stats import spearmanr

## lasso
from sklearn.linear_model import Lasso
# return lasso filtered freatures based on training set.
def go_lasso(X, y, alpha):
	lassoFS = Lasso(alpha = alpha)
	lassoFS.fit(X, y)
	coefs = lassoFS.coef_
	fts_ids = [int(i) for i, coef in enumerate(coefs) if coef!=0]
	return fts_ids

## main

## has CV
if cvFold > 0:
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=cvFold, random_state=random_state, shuffle=True)
	kf.get_n_splits(X)

	s1s, s2s = [], []
	preds, Ys = [], []
	for train_index, test_index in kf.split(X):
		X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		if lasso:
			fts_ids = go_lasso(X_train, y_train, alpha = lasso)
			X_train = X_train.iloc[:,fts_ids]
			X_test = X_test.iloc[:,fts_ids]
		model.fit(X_train, y_train)
		pred = model.predict(X_test)
		if mode == "classification" or "c" or "C":
			predp = model.predict_proba(X_test)[:,1]
			auROC = roc_auc_score(y_test, predp)
			auPRC = average_precision_score(y_test, predp)
			print("ROC\t{0:.3f}\tPRC\t{1:.3f}".format(auROC, auPRC))
			s1s.append(auROC)
			s2s.append(auPRC)
		else:
			#spcor, pval = spearmanr(y_test, pred)
			r2 = r2_score(y_test, pred)
			varExp = explained_variance_score(y_test, pred)
			#print("spearman\t{}\tvarExp\t{}".format(spcor, varExp))
			print("r2\t{}\tvarExp\t{}".format(r2, varExp))
			#s1s.append(spcor)
			s1s.append(r2)
			s2s.append(varExp)
		# for predFile
		preds += list(pred)
		Ys += list(y_test)
	print("{}\t{}\t{}\t{}".format(round(mean(s1s),3), round(std(s1s),3), round(mean(s2s),3), round(std(s2s),3)))

## no CV; regardless, full model is computed, imp should come from here. 
if lasso:
	fts_ids = go_lasso(X, y, alpha = lasso)
	X = X.iloc[:,fts_ids]
	fts = [fts[fts_id] for fts_id in fts_ids]
modelFull = model
modelFull.fit(X, y)
predFull = modelFull.predict(X)
## importance
## print imp
def getImp(model, fts):
	lines = []
	for i, e in enumerate(sorted(list(zip(model.feature_importances_, fts)), reverse=True)):
		line = "\t".join(map(str,e))
		lines.append(line)
	return "\n".join(lines)
## write imp
with open(impFile, "w") as f:
	f.write(getImp(modelFull, fts))

## otherwise print full model performance.
if cvFold <= 0:
	if mode == "classification" or "c" or "C":
		predp = modelFull.predict_proba(X)[:,1]
		auROC = roc_auc_score(y, predp)
		auPRC = average_precision_score(y, predp)
		print("ROC\t{0:.3f}\tPRC\t{1:.3f}".format(auROC, auPRC))
	else:
		r2 = r2_score(y, predFull)
		varExp = explained_variance_score(y, predFull)
		print("r2\t{}\tvarExp\t{}".format(r2, varExp))
	# for predFile
	Ys = y
	preds = predFull

## pred-Y
with open(predFile, "w") as f:
	f.write("\n".join(["{}\t{}".format(pair[0], pair[1]) for pair in zip(Ys, preds)]))

