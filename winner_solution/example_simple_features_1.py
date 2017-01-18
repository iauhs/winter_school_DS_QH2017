#coding:utf-8

import random
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn import preprocessing
from scipy.spatial.distance import cosine
import nltk
import time
import networkx as nx
import csv

#nltk.download('punkt') # for tokenization
#nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

with open("testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set  = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]
# data loading and preprocessing

# the columns of the data frame below are:
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

with open("training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set  = list(reader)
#to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.5)))
#to_keep=len(training_set)
#从1-training——set中随机选择sample range(5) 从0-5选数，不包括5 选0.05的数据作为样本
#training_set = [training_set[i] for i in to_keep]

training_set = [element[0].split(" ") for element in training_set]
with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)

print 'print graph'
graph = nx.Graph()
edges = []
nodes = set()
for i in xrange(len(training_set)):
    source = training_set[i][0]
    target = training_set[i][1]
    nodes.add(source)
    nodes.add(target)
    if training_set[i][2]=='1':
        edges.append((source,target))

graph.add_nodes_from(nodes)
graph.add_edges_from(edges)

neighbor_count_source = []
neighbor_count_target = []

degree_set = graph.degree()
core_set = nx.core_number(graph)
cluster_set = nx.clustering(graph)
page_set = nx.pagerank(graph)
print 'Finished Graph'

#to test code we select sample
valid_ids=set()
for element in training_set:
    valid_ids.add(element[0])
    valid_ids.add(element[1])

tmp=[element for element in node_info if element[0] in valid_ids ]
node_info=tmp
del tmp



IDs = []
ID_pos={}
for element in node_info:
	ID_pos[element[0]]=len(IDs)
	IDs.append(element[0])

corpus=[elements[5] for elements in node_info]
vectorizer=TfidfVectorizer(stop_words='english',max_df=0.01)

M=vectorizer.fit_transform(corpus)

# we will use three basic features:
training_features=[]
# number of overlapping words in title
overlap_title = []
overlap_abs = []

# temporal distance between the papers
temp_diff = []

# number of common authors
comm_auth = []
overlap_jour=[]
tfidt_cos=[]
# graph related
neighbor_sim=[]
degree_1=[]
degree_2=[]
core_1=[]
core_2=[]
cluster_1=[]
cluster_2=[]
pagerank_1=[]
pagerank_2=[]
delta_pagerank=[]
delat_cluster=[]
delta_core=[]
delta_degree=[]


token_valid={'NN','JJ','VB','CD','JJR','JJS','NNP','NNS','VBD','VBZ','VBN','VBP'}
time0 = time.time()
#counter = 0
for i in xrange(len(training_set)):
    source = training_set[i][0]
    target = training_set[i][1]

    source_degree = degree_set[source]
    target_degree = degree_set[target]

    source_core = core_set[source]
    target_core = core_set[target]

    source_cluster = cluster_set[source]
    target_cluster = cluster_set[target]

    source_page = page_set[source]
    target_page = page_set[target]

    source_info = node_info[ID_pos[source]]
    target_info = node_info[ID_pos[target]]

	#about title
    source_title_tokened = nltk.word_tokenize(source_info[2].lower())
    target_title_tokened = nltk.word_tokenize(target_info[2].lower())
    source_title_tagged = nltk.pos_tag(source_title_tokened)
    target_title_tagged = nltk.pos_tag(target_title_tokened)
	 # remove stopwords and not verbs and nones
    source_title = [token[0] for token in source_title_tagged if token[0] not in stpwds and token[1] in token_valid]
    source_title = [stemmer.stem(token) for token in source_title]
    target_title = [token[0] for token in target_title_tagged if token[0] not in stpwds and token[1] in token_valid]
    target_title = [stemmer.stem(token) for token in target_title]

#    # about abstract
    source_abs_tokened = nltk.word_tokenize(source_info[5].lower())
    target_abs_tokened = nltk.word_tokenize(target_info[5].lower())
#    source_abs_tagged = nltk.pos_tag(source_abs_tokened)
#    target_abs_tagged = nltk.pos_tag(target_abs_tokened)
#	# remove stopwords and not verbs and nones
    source_abs = [token[0] for token in source_abs_tokened if token[0] not in stpwds]
    source_abs = [stemmer.stem(token) for token in source_abs]
    target_abs = [token[0] for token in target_abs_tokened if token[0] not in stpwds]
    target_abs = [stemmer.stem(token) for token in target_abs]
#
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")

    source_jour = source_info[4].lower()
    target_jour = target_info[4].lower()

    if len(source_jour)!=0 and len(target_jour)!=0:
        overlap_jour.append(len(set([source_jour]).intersection(set([target_jour]))))
    else:
        overlap_jour.append(0)

    vector_1=M[ID_pos[source],:].toarray()[0]
    vector_2=M[ID_pos[target],:].toarray()[0]
    if np.linalg.norm(vector_1)!=0 and np.linalg.norm(vector_2)!=0:
        temp_cosine=cosine(vector_1,vector_2)
    else:
        temp_consin=0.0

    neighbor_source=len(list(graph.neighbors(source)))
    neighbor_target=len(list(graph.neighbors(target)))
    neighbor_count_source.append(neighbor_source)
    neighbor_count_target.append(neighbor_target)
    neighbor_source=set(list(graph.neighbors(source)))
    neighbor_target=set(list(graph.neighbors(target)))

    tfidt_cos.append(temp_cosine)
    overlap_abs.append(len(set(source_abs).intersection(set(target_abs))))
    overlap_title.append(len(set(source_title).intersection(set(target_title))))
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
    degree_1.append(source_degree)
    degree_2.append(target_degree)
    core_1.append(source_core)
    core_2.append(target_core)
    cluster_1.append(source_cluster)
    cluster_2.append(target_cluster)
    pagerank_1.append(source_page)
    pagerank_2.append(target_page)
    neighbor_sim.append(len(neighbor_source.intersection(neighbor_target)))

    if i % 1000 == 0:
        time1=time.time()
        print i, "training examples processsed elapsed", time1-time0
	#counter = counter + 1

# convert list of lists into array
delta_pagerank=(np.array(pagerank_1)-np.array(pagerank_2)).tolist()
delat_cluster=(np.array(cluster_1)-np.array(cluster_2)).tolist()
delta_core=(np.array(core_1)-np.array(core_2)).tolist()
delta_degree=(np.array(degree_1)-np.array(degree_2)).tolist()
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
training_features = np.array([overlap_title, temp_diff, comm_auth, tfidt_cos,neighbor_sim,overlap_jour,overlap_abs,
                              delta_pagerank,delat_cluster,delta_core,delta_degree,
                              degree_1,degree_2,core_1,core_2,cluster_1,cluster_2,pagerank_1,pagerank_2]).T

# scale
training_features = preprocessing.scale(training_features)

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set]
labels = list(labels)
labels_array = np.array(labels)


# ******************************** test process ***************************************
# number of overlapping words in title
overlap_title = []
overlap_abs = []
# temporal distance between the papers
temp_diff = []
# number of common authors
comm_auth = []
overlap_jour=[]
tfidt_cos=[]
# graph related
neighbor_sim=[]
degree_1=[]
degree_2=[]
core_1=[]
core_2=[]
cluster_1=[]
cluster_2=[]
pagerank_1=[]
pagerank_2=[]
delta_pagerank=[]
delat_cluster=[]
delta_core=[]
delta_degree=[]

for i in xrange(len(testing_set)):
    source = testing_set[i][0]
    target = testing_set[i][1]

    source_degree = degree_set[source]
    target_degree = degree_set[target]

    source_core = core_set[source]
    target_core = core_set[target]

    source_cluster = cluster_set[source]
    target_cluster = cluster_set[target]

    source_page = page_set[source]
    target_page = page_set[target]

    source_info = node_info[ID_pos[source]]
    target_info = node_info[ID_pos[target]]

	#about title
    source_title_tokened = nltk.word_tokenize(source_info[2].lower())
    target_title_tokened = nltk.word_tokenize(target_info[2].lower())
    source_title_tagged = nltk.pos_tag(source_title_tokened)
    target_title_tagged = nltk.pos_tag(target_title_tokened)
	 # remove stopwords and not verbs and nones
    source_title = [token[0] for token in source_title_tagged if token[0] not in stpwds and token[1] in token_valid]
    source_title = [stemmer.stem(token) for token in source_title]
    target_title = [token[0] for token in target_title_tagged if token[0] not in stpwds and token[1] in token_valid]
    target_title = [stemmer.stem(token) for token in target_title]

#    # about abstract
    source_abs_tokened = nltk.word_tokenize(source_info[5].lower())
    target_abs_tokened = nltk.word_tokenize(target_info[5].lower())
#    source_abs_tagged = nltk.pos_tag(source_abs_tokened)
#    target_abs_tagged = nltk.pos_tag(target_abs_tokened)
#	# remove stopwords and not verbs and nones
    source_abs = [token[0] for token in source_abs_tokened if token[0] not in stpwds]
    source_abs = [stemmer.stem(token) for token in source_abs]
    target_abs = [token[0] for token in target_abs_tokened if token[0] not in stpwds]
    target_abs = [stemmer.stem(token) for token in target_abs]
#
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")

    source_jour = source_info[4].lower()
    target_jour = target_info[4].lower()

    if len(source_jour)!=0 and len(target_jour)!=0:
        overlap_jour.append(len(set([source_jour]).intersection(set([target_jour]))))
    else:
        overlap_jour.append(0)

    vector_1=M[ID_pos[source],:].toarray()[0]
    vector_2=M[ID_pos[target],:].toarray()[0]
    if np.linalg.norm(vector_1)!=0 and np.linalg.norm(vector_2)!=0:
        temp_cosine=cosine(vector_1,vector_2)
    else:
        temp_consin=0.0

    neighbor_source=len(list(graph.neighbors(source)))
    neighbor_target=len(list(graph.neighbors(target)))
    neighbor_count_source.append(neighbor_source)
    neighbor_count_target.append(neighbor_target)
    neighbor_source=set(list(graph.neighbors(source)))
    neighbor_target=set(list(graph.neighbors(target)))

    tfidt_cos.append(temp_cosine)
    overlap_abs.append(len(set(source_abs).intersection(set(target_abs))))
    overlap_title.append(len(set(source_title).intersection(set(target_title))))
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
    degree_1.append(source_degree)
    degree_2.append(target_degree)
    core_1.append(source_core)
    core_2.append(target_core)
    cluster_1.append(source_cluster)
    cluster_2.append(target_cluster)
    pagerank_1.append(source_page)
    pagerank_2.append(target_page)
    neighbor_sim.append(len(neighbor_source.intersection(neighbor_target)))

    if i % 1000 == 0:
        time1=time.time()
        print i, "testing examples processsed elapsed", time1-time0
        #counter += 1

delta_pagerank=(np.array(pagerank_1)-np.array(pagerank_2)).tolist()
delat_cluster=(np.array(cluster_1)-np.array(cluster_2)).tolist()
delta_core=(np.array(core_1)-np.array(core_2)).tolist()
delta_degree=(np.array(degree_1)-np.array(degree_2)).tolist()

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
training_features_1 = np.array([overlap_title, temp_diff, comm_auth, tfidt_cos,neighbor_sim,overlap_jour,overlap_abs,
                              delta_pagerank,delat_cluster,delta_core,delta_degree,
                              degree_1,degree_2,core_1,core_2,cluster_1,cluster_2,pagerank_1,pagerank_2]).T

# scale
training_features_1 = preprocessing.scale(training_features_1)

# convert labels into integers then into column array

print "evaluating"


#evaluation
#kf = KFold(len(training_set), n_folds=10)
#sumf1=0

#for train_index, test_index in kf:
#	X_train, X_test = training_features[train_index], training_features[test_index]
#	y_train, y_test = labels_array[train_index], labels_array[test_index]
	# initialize basic SVM
#	classifier = svm.LinearSVC()
	# train
#	classifier.fit(X_train, y_train)
#	pred=classifier.predict(X_test)
#	sumf1+=f1_score(pred,y_test)

def gradient_boosting_classifier(training_features, labels_array):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(training_features, labels_array)
    return model


#def knn_classifier(train_x, train_y):
#    from sklearn.neighbors import KNeighborsClassifier
#    model = KNeighborsClassifier()
#    model.fit(train_x, train_y)
#    return model

#model = knn_classifier(training_features, labels_array)
#predict = model.predict(training_features_1)


print '**************************************'
#start_time = time.time()
model = gradient_boosting_classifier(training_features, labels_array)
#print 'training took %fs!' % (time.time() - start_time)
predict = model.predict(training_features_1)


with open('prediction_net_xuejie_19features.csv','wb') as sub:
    csv_out=csv.writer(sub)
    csv_out.writerow(['ID','prediction'])
    for a,b in enumerate(predict):
        csv_out.writerow([a,b])
    sub.close()

#classifier = svm.LinearSVC()
#classifier.fit(training_features, labels_array)
#pred = classifier.predict(training_features_1)
#
#with open('new_predictions_net_xuejie.csv','wb') as sub:
#    csv_out=csv.writer(sub)
#    csv_out.writerow(['overlap_title', 'overlap_abs', 'temp_diff', 'comm_auth', 'tfidt_cos'])
#    for index,row in enumerate(predict):
#        csv_out.writerow([index,row])
#    sub.close()
#
#print "\n\n"
#print sumf1/10.0

#with open('overlap_abs_training_features.csv','wb') as sub:
#    csv_out=csv.writer(sub)
#    csv_out.writerow(['overlap_title', 'overlap_abs', 'temp_diff', 'comm_auth', 'tfidt_cos'])
#    for a,b,c,d,e in training_features:
#        csv_out.writerow([a,b,c,d,e])
#    sub.close()
#
#with open('overlap_abs_training_features_ori.csv','wb') as sub:
#    csv_out=csv.writer(sub)
#    csv_out.writerow(['overlap_title', 'overlap_abs', 'temp_diff', 'comm_auth', 'tfidt_cos'])
#    for a,b,c,d,e in training_features:
#        csv_out.writerow([a,b,c,d,e])
#    sub.close()
#
#with open('overlap_abs_testing_features.csv','wb') as sub:
#    csv_out=csv.writer(sub)
#    csv_out.writerow(['overlap_title', 'overlap_abs', 'temp_diff', 'comm_auth', 'tfidt_cos'])
#    for a,b,c,d,e in training_features_1:
#        csv_out.writerow([a,b,c,d,e])
#    sub.close()
#
#with open('overlap_abs_training_features_ori.csv','wb') as sub:
#    csv_out=csv.writer(sub)
#    csv_out.writerow(['overlap_title', 'overlap_abs', 'temp_diff', 'comm_auth', 'tfidt_cos'])
#    for a,b,c,d,e in np.array([overlap_title, overlap_abs, temp_diff, comm_auth, tfidt_cos]).T:
#        csv_out.writerow([a,b,c,d,e])
#    sub.close()
#
#with open('token.csv','wb') as sub:
#    csv_out=csv.writer(sub)
#    csv_out.writerow(['overlap_title', 'overlap_abs', 'temp_diff', 'comm_auth', 'tfidt_cos'])
#    for a,b,c,d,e in np.array([source_title_tokened, overlap_abs, temp_diff, comm_auth, tfidt_cos]).T:
#        csv_out.writerow([a,b,c,d,e])
#    sub.close()
