#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 01:42:21 2017

@author: vincent
"""


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random
import numpy as np
import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn import preprocessing
from scipy.spatial.distance import cosine
import nltk
import csv
import networkx as nx


nltk.download('punkt')  # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

with open("../input/testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]
# data loading and preprocessing

# the columns of the data frame below are:
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

with open("../input/training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set = list(reader)

training_set = [element[0].split(" ") for element in training_set]
with open("../input/node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info = list(reader)

# to test code we select sample
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set) * 0.05)))
#to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set))))
training_set = [training_set[i] for i in to_keep]
print(len(training_set))
valid_ids = set()
for element in training_set:
    valid_ids.add(element[0])
    valid_ids.add(element[1])

tmp = [element for element in node_info if element[0] in valid_ids]
node_info = tmp
del tmp

IDs = []
ID_pos = {}
for element in node_info:
    ID_pos[element[0]] = len(IDs)
    IDs.append(element[0])

print ("build graph")
graph = nx.Graph()
edges = []
nodes = set()


for i in range(len(training_set)):
    source = training_set[i][0]
    target = training_set[i][1]
    nodes.add(source)
    nodes.add(target)
    if training_set[i][2]=="1":
        edges.append((source,target))

graph.add_nodes_from(nodes)
graph.add_edges_from(edges)

neighbor_count_source = []
neighbor_count_target = []



# we will use three basic features:

# number of overlapping words in title
overlap_title = []

# temporal distance between the papers
temp_diff = []

# number of common authors
comm_auth = []


#similarity based on neighbors in graph
neighbor_sim=[]



#ifidf similarities
tfidf_cos=[ ]
corpus=[elements[5] for elements in node_info]
vectorizer = TfidfVectorizer(stop_words="english",max_df=0.01)
M=vectorizer.fit_transform(corpus)


counter = 0
for i in range(len(training_set)):
    source = training_set[i][0]
    target = training_set[i][1]

    source_info = node_info[ID_pos[source]]
    target_info = node_info[ID_pos[target]]

    # convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
    # remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    source_title = [stemmer.stem(token) for token in source_title]

    target_title = target_info[2].lower().split(" ")
    target_title = [token for token in target_title if token not in stpwds]
    target_title = [stemmer.stem(token) for token in target_title]

    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")

    vector1 = M[ID_pos[source], :].toarray()[0]
    vector2 = M[ID_pos[target], :].toarray()[0]
    temp_cosine=0.0
    if np.linalg.norm(vector1)!=0 and np.linalg.norm(vector2)!=0:
        temp_cosine=cosine(vector1,vector2)

    tfidf_cos.append(temp_cosine)
    neighbor_source = len(list(graph.neighbors(source)))
    neighbor_target = len(list(graph.neighbors(target)))
    neighbor_count_source.append(neighbor_source)
    neighbor_count_target.append(neighbor_target)
    neighbor_source = set(list(graph.neighbors(source)))
    neighbor_target = set(list(graph.neighbors(target)))
    neighbor_sim.append(len(neighbor_source.intersection(neighbor_target)))



    overlap_title.append(len(set(source_title).intersection(set(target_title))))
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))

    if counter % 10000 == 0:
        print(counter, "training examples processsed")
    counter += 1

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
#training_features = np.array([overlap_title, temp_diff, comm_auth, tfidf_cos, neighbor_sim]).T
training_features = np.array([overlap_title, temp_diff, comm_auth, neighbor_sim,neighbor_count_source,neighbor_count_target,tfidf_cos]).T
print(training_features)

# scale
training_features = preprocessing.scale(training_features)

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set]
labels = list(labels)
labels_array = np.array(labels)

print("evaluating")

# evaluation
kf = KFold(len(training_set), n_folds=10)
sumf1 = 0
for train_index, test_index in kf:
    X_train, X_test = training_features[train_index], training_features[test_index]
    y_train, y_test = labels_array[train_index], labels_array[test_index]
    # initialize basic SVM
    classifier = svm.LinearSVC()
    # train
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    sumf1 += f1_score(pred, y_test)

print("\n\n")
print(sumf1 / 10.0)


