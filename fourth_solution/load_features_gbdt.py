import random
import numpy as np
#import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn import preprocessing
from scipy.spatial.distance import cosine
import nltk
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn import decomposition
import sklearn.feature_extraction.text as text
import networkx as nx
from scipy.sparse.linalg import svds
from sklearn.ensemble import GradientBoostingClassifier
from getIDS import *

# print 'load_dataset'
# with open("training_set.txt", "r") as f:
#         reader = csv.reader(f)
#         data_set = list(reader)

# data_set = [element[0].split(" ") for element in data_set]

print 'training_features'
training_features = np.loadtxt(open("training_feautres.csv","rb"),delimiter=",",skiprows=0)

print 'training_labels'
training_labels = np.loadtxt(open("training_labels.csv","rb"),delimiter=",",skiprows=0)

print 'trest_features'
test_features = np.loadtxt(open("test_features.csv","rb"),delimiter=",",skiprows=0)

print 'getIDS'
IDs = getIDS()

# classifier = RandomForestClassifier(n_estimators=300)
# classifier.fit(training_features, training_labels)
# pred=classifier.predict(test_features)
# results = np.concatenate((np.reshape(IDs,(np.size(pred), 1)), np.reshape(pred, (np.size(pred),1))), axis=1)
# with open("new_submission.txt_rf", "w") as f:
#     writer = csv.writer(f, )
#     writer.writerow(['id','prediction'])
#     writer.writerows(results)     


# from sklearn import datasets
# from sklearn import cross_validation
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import VotingClassifier

# iris = datasets.load_iris()
# X, y = iris.data[:, 1:3], iris.target

# clf1 = LogisticRegression(random_state=1)
# clf2 = RandomForestClassifier(random_state=1)
# clf3 = GaussianNB()

# eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard', weights=[2,1,2])
# eclf.fit(training_features,training_labels)
# eclf.fit(test_features)
# results = np.concatenate((np.reshape(IDs,(np.size(pred), 1)), np.reshape(pred, (np.size(pred),1))), axis=1)
# with open("new_submission_vote.txt", "w") as f:
#     writer = csv.writer(f, )
#     writer.writerow(['id','prediction'])
#     writer.writerows(results)  


classifier=GradientBoostingClassifier()
classifier.fit(training_features,training_labels)
pred=classifier.predict(test_features)

# results = np.concatenate((np.reshape(IDs,(np.size(pred), 1)), np.reshape(pred, (np.size(pred),1))), axis=1)
print 'write_result'
with open("new_submission_gbdt.txt", "w") as f:
    # writer = csv.writer(f, )
    # writer.writerow(['id','prediction'])
    # writer.writerows(results)  
    csv_out=csv.writer(f)
    csv_out.writerow(['id','prediction'])
    for index,row in enumerate(pred):
        csv_out.writerow([index,int(row)])




#xgboost
# import xgboost as xgb
# param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
# dtrain = xgb.DMatrix("../output/training_feautres.csv", label="../output/training_labels.csv")
# dtrain.save_binary("train.buffer")
# dtest = xgb.DMatrix(test_features)
# pred = bst.predict(dtest)
# bst = xgb.train( param, dtrain, num_round, evallist )
# results = np.concatenate((np.reshape(IDs,(np.size(pred), 1)), np.reshape(pred, (np.size(pred),1))), axis=1)
# with open("new_submission_gbdt.txt", "w") as f:
#     writer = csv.writer(f, )
#     writer.writerow(['id','prediction'])
#     writer.writerows(results)  


# #evaluate

# print 'evaluate'
# kf = KFold(len(data_set), n_folds=10)
# sumf1=0
# for train_index, test_index in kf:
#     X_train, X_test = training_features[train_index], training_features[test_index]
#     y_train, y_test = training_labels[train_index], training_labels[test_index]
#     # initialize basic SVM
#     # classifier = svm.LinearSVC()
#     # classifier = svm.SVC()
#     # # train
#     # classifier.fit(X_train, y_train)
#     classifier=GradientBoostingClassifier()
#     classifier.fit(X_train,y_train)   
#     pred=classifier.predict(X_test)
#     sumf1+=f1_score(pred,y_test)

# print "\n\n"
# print sumf1/10.0