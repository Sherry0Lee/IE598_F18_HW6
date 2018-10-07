#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 21:11:33 2018

@author: sherry
"""

from sklearn import datasets
import numpy as np
from sklearn import metrics

score_save=[]
iris=datasets.load_iris()
x=iris.data[:,[2,3]]
y=iris.target
print('Class labels:',np.unique(y))

# Random Test Train Splits
for i in range(10):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=i,stratify=y)
    print('Labels counts in y_train:',np.bincount(y_train))
    print('Labels counts in y_test:',np.bincount(y_test))
       
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
    tree.fit(x_train,y_train)
    y_train_pred=tree.predict(x_train)
    y_pred=tree.predict(x_test)
    score_in=metrics.accuracy_score(y_train,y_train_pred)
    score_out=metrics.accuracy_score(y_test,y_pred)
    print(score_in,score_out)
    score_save.append(score_out)
    
print(score_save)
print("\nmean scores:",np.mean(score_save))
print("Standard Deviation of the scores:",np.std(score_save))

# Cross Validation
from sklearn.model_selection import cross_val_score
for i in range(10):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=i,stratify=y)
    cv_scores=cross_val_score(estimator=tree,X=x_train,y=y_train,cv=10,n_jobs=1)
    print("\nCV accuracy scores:%s"%cv_scores)
    print("CV accuracy:%.3f +/- %.3f"%(np.mean(cv_scores),np.std(cv_scores)))
    tree.fit(x_train,y_train)
    y_pred=tree.predict(x_test)
    cv_score_out=metrics.accuracy_score(y_test,y_pred)
    print("out of sample accuracy:",cv_score_out) 

print("My name is Sihan Li")
print("My NetID is: sihanl2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
