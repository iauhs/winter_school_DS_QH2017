#coding:utf-8

import random
import numpy as np
import igraph

import networkx as nx
#import matplotlib.pyplot as plt
#from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn import preprocessing
from scipy.spatial.distance import cosine
import nltk
import csv
#from sklearn.cluster import KMeans 
#from gensim.models import Word2Vec  
#from gensim.models.word2vec import LineSentence  
#import multiprocessing
#import os
#mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
#os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
#import xgboost as xgb
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

training_set = [element[0].split(" ") for element in training_set]
with open("node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info  = list(reader)
    
#testing_Set_1=[element[0].split(" ") for element in testing_Set]

#to test code we select sample
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.05)))
#to_keep=len(training_set)
#从1-training——set中随机选择sample range(5) 从0-5选数，不包括5 选0.05的数据作为样本
training_set = [training_set[i] for i in to_keep]
valid_ids=set()
#定义一个集合
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

# community.
# codes add for LSA ##
print 'computing LSA'
## compute TFIDF vectr of each paper
corpus= [element[5] for element in node_info]
vectorizer=TfidfVectorizer(stop_words='english')
#each row is a node in the orfer of node_info
features_TFIDF=vectorizer.fit_transform(corpus) 

## lSA --keep part of the document for memory reasons (around 2K rows)
corpus2=[corpus[i] for i in sorted(random.sample(xrange(len(corpus)),k=2000))]
A=vectorizer.transform(corpus2).toarray()
## apply singular value decomposition
U, S, V=np.linalg.svd(A)
## keep the first 4 rows of V
V2=V[:100,:] #can try other vlaue, conoptimize 

## the matirx after dimensionality reduction
M = np.dot(features_TFIDF.toarray(),V2.transpose())

print "LSA Done"

#
#from igraph import *
#g = Graph()
#for element in node_info:
#    g.add_vertices(element[0])

#g.add_edges([(0,1), (1,2)])
#for element in training_set:
#    node1_temp=ID_pos[element[0]]
#    node2_temp=ID_pos[element[1]]
#    if element[2]=='1':  
#        g.add_edges([(node1_temp,node2_temp)])
    #a=element

#tfidf simularities
#corpus=[elements[]]
#abs_tokened=[nltk.word_tokenize(elements[5].lower()) for elements in node_info]
#abs_tagged =[nltk.pos_tag(element) for element in abs_tokened]

corpus=[elements[2] for elements in node_info]
vectorizer=TfidfVectorizer(stop_words='english',max_df=0.01)
M1=vectorizer.fit_transform(corpus)
#clf = KMeans(n_clusters=20)  
#s = clf.fit(temp_consin)  
# we will use three basic features:

training_features=[]
# number of overlapping words in title
overlap_title = []
overlap_abs = []

# temporal distance between the papers
temp_diff = []
temp1= []
temp2= []


# number of common authors
comm_auth = []

# LSA 
LSA = []


tfidt_cos=[]
overlap_jour=[]
#similarity based on neighbors in graph
neighbor_sim=[]
degree_1=[]
degree_2=[]
core_1=[]
core_2=[]
cluster_1=[]
cluster_2=[]
pagerank_1=[]
pagerank_2=[]


token_valid={'NN','JJ','VB','CD','JJR','JJS','NNP','NNS','VBD','VBZ','VBN','VBP'}


#[degree_set]

counter = 0
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
#    source_abs_tokened = nltk.word_tokenize(source_info[5].lower())
#    target_abs_tokened = nltk.word_tokenize(target_info[5].lower())   
#    source_abs_tagged = nltk.pos_tag(source_abs_tokened)
#    target_abs_tagged = nltk.pos_tag(target_abs_tokened)
#	# remove stopwords and not verbs and nones   
#    source_abs = [token[0] for token in source_abs_tagged if token[0] not in stpwds and token[1] in token_valid]
#    source_abs = [stemmer.stem(token) for token in source_abs]    
#    target_abs = [token[0] for token in target_abs_tagged if token[0] not in stpwds and token[1] in token_valid]
#    target_abs = [stemmer.stem(token) for token in target_abs]
    
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")
    
    source_jour = source_info[4].lower()
    target_jour = target_info[4].lower()
    
    vector_1=M1[ID_pos[source],:].toarray()[0]
    vector_2=M1[ID_pos[target],:].toarray()[0]
    if np.linalg.norm(vector_1)!=0 and np.linalg.norm(vector_2)!=0:
        temp_cosine=cosine(vector_1,vector_2)
    else:
        temp_consin=0.0
    tfidt_cos.append(temp_cosine)    
#    overlap_abs.append(len(set(source_abs).intersection(set(target_abs))))
    overlap_title.append(len(set(source_title).intersection(set(target_title))))
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    temp1.append(int(source_info[1]))
    temp2.append(int(target_info[1]))

    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))  
    degree_1.append(source_degree)
    degree_2.append(target_degree)
    core_1.append(source_core)
    core_2.append(target_core)
    cluster_1.append(source_cluster)
    cluster_2.append(target_cluster)
    pagerank_1.append(source_page)
    pagerank_2.append(target_page)
    
    neighbor_source=len(list(graph.neighbors(source)))
    neighbor_target=len(list(graph.neighbors(target)))
    neighbor_count_source.append(neighbor_source)
    neighbor_count_target.append(neighbor_target)  
    neighbor_source=set(list(graph.neighbors(source)))
    neighbor_target=set(list(graph.neighbors(target)))
        
    neighbor_sim.append(len(neighbor_source.intersection(neighbor_target)))
    
    LSA.append(cosine(M[ID_pos[target],:],M[ID_pos[source],:]))

    
    if len(source_jour)!=0 and len(target_jour)!=0:
        overlap_jour.append(len(set([source_jour]).intersection(set([target_jour]))))
    else:
        overlap_jour.append(0)    
     
    if i % 1000 == 0:
        print i, "training examples processsed"  
        
    
delta_pagerank=(np.array(pagerank_1)-np.array(pagerank_2)).tolist()
delat_cluster=(np.array(cluster_1)-np.array(cluster_2)).tolist()
delta_core=(np.array(core_1)-np.array(core_2)).tolist()
delta_degree=(np.array(degree_1)-np.array(degree_2)).tolist()


#watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]
#evallist  = [(dtest,'eval'), (dtrain,'train')]		
# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
#training_features = np.array([pagerank_1,pagerank_2,cluster_1,cluster_2,core_1, core_2, degree_1, degree_2, neighbor_sim, overlap_jour, overlap_title, temp_diff, comm_auth, tfidt_cos]).T
#training_features = np.array([LSA,pagerank_1,pagerank_2,cluster_1,cluster_2,core_1, core_2, degree_1, degree_2, neighbor_sim, overlap_jour, overlap_title, temp_diff, comm_auth, tfidt_cos]).T
training_features = np.array([LSA,pagerank_1,pagerank_2,cluster_1,cluster_2,
                              core_1, core_2, degree_1, degree_2, neighbor_sim,
                              overlap_jour, overlap_title, temp1,temp2,temp_diff,
                              comm_auth, tfidt_cos]).T

#training_features = np.array([delta_pagerank,delat_cluster,delta_core,delta_degree,overlap_title, comm_auth, tfidt_cos,neighbor_sim,overlap_jour,temp_diff]).T
#training_features = np.array([comm_auth, tfidt_cos,neighbor_sim,overlap_jour,temp_diff]).T

# scale
training_features = preprocessing.scale(training_features)

# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set]
labels = list(labels)
labels_array = np.array(labels)

print "evaluating"


#params={
#'booster':'gbtree',
#'objective': 'binary:logistic', #多分类的问题
#'num_class':10, # 类别数，与 multisoftmax 并用
#'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
#'max_depth':12, # 构建树的深度，越大越容易过拟合
#'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#'subsample':0.7, # 随机采样训练样本
#'colsample_bytree':0.7, # 生成树时进行的列采样
#'min_child_weight':3, 
## 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
##，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
##这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
#'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
#'eta': 0.02, # 如同学习率
#'seed':1000,
#'nthread':7,# cpu 线程数
##'eval_metric': 'auc'
#}
#plst = list(params.items())
#num_round = 10
#dtrain = xgb.DMatrix(training_features, label=labels_array)
##dtest = xgb.DMatrix(training_features_1)
#bst = xgb.train( plst, dtrain, num_round, evallist )

#for i in range(300,300,10):
def gradient_boosting_classifier(training_features, labels_array):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier(n_estimators=300,loss='deviance')  
    model.fit(training_features, labels_array)  
    return model
    
#    dtrain = xgb.Dmatrix()  
      
    #evaluation
kf = KFold(len(training_set), n_folds=10)
sumf1=0
for train_index, test_index in kf:
	X_train, X_test = training_features[train_index], training_features[test_index]
	y_train, y_test = labels_array[train_index], labels_array[test_index]
	# initialize basic SVM
#	dtrain = xgb.DMatrix(X_train, label=y_train)
	#evallist=
	#bst = xgb.train( plst, dtrain, num_round, evallist )
	#classifier = gradient_boosting_classifier(training_features, labels_array)  
#print 'training took %fs!' % (time.time() - start_time)  
	classifier = gradient_boosting_classifier(training_features, labels_array)
	# train
	#classifier.fit(X_train, y_train)
	pred=classifier.predict(X_test)
	sumf1+=f1_score(pred,y_test)
print i, sumf1/10.0


#with open('diff.csv','wb') as sub:
#    csv_out=csv.writer(sub)
#    #csv_out.writerow(['ID','prediction'])
#    for a in temp_diff:
#      csv_out.writerow([a])
    
      

