import random
import numpy as np
import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn import preprocessing
from scipy.spatial.distance import cosine
import nltk
import csv
import networkx as nx

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



#to test code we select sample
# to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.05)))
# training_set = [training_set[i] for i in to_keep]
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


#building graph
G = nx.DiGraph()
G.add_edges_from([(elem[0],elem[1]) for elem in training_set if elem[2] == 1])
edges=[]

#building graph2
graph = nx.Graph()
edges=[]
nodes=set()
for i in xrange(len(training_set)):
    source = training_set[i][0]
    target = training_set[i][1]
    nodes.add(source)
    nodes.add(target)
    if training_set[i][2] == "1":
        edges.append((source,target))
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)

neighbor_count_source=[]
neighbor_count_target=[]

#similarity based on neighbors
# neighbor_count_source.append(len(list(graph.neighbors(source))))
# neighbor_count_target.append(len(list(graph.neighbors(target))))

neighbor_sim=[]

#cluster nodes
cluster_nodes=[]
# we will use three basic features:



# number of overlapping words in title
overlap_title = []

# temporal distance between the papers
temp_diff = []

# number of common authors
comm_auth = []

#content text feauture
content_fea=[]

#tfidf similarities
tfidf_cos=[]
vectorizer=TfidfVectorizer(stop_words="english")
# vectorizer=HashingVectorizer(n_features=(100))

corpus=[element[5] for element in node_info]

# M = vectorizer.fit_transform(corpus)
features_TFIDF = vectorizer.fit_transform(corpus)

corpus2=[corpus[i] for i in sorted(random.sample(xrange(len(corpus)),k=2000)) ]
A = vectorizer.transform(corpus2).toArray()
U,S,V = np.linalg.svd(A)
V2 = V[:100,:]
M = np.dot(features_TFIDF.toarray(),V2.transpose())

print "LSA DONE"


counter = 0
for i in xrange(len(training_set)):
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

    source_sim = M[ID_pos[source],:].toarray()[0]
    target_sim = M[ID_pos[target],:].toarray()[0]

    temp_cosine = 0.0
    if np.linalg.norm(source_sim)!=0 and np.linalg.norm(target_sim)!=0 :
        temp_cosine = cosine(source_sim,target_sim)
    tfidf_cos.append(temp_cosine)


    neighbor_source = len(list(graph.neighbors(source)))
    neighbor_target = len(list(graph.neighbors(target)))
    neighbor_count_source.append(neighbor_source)
    neighbor_count_target.append(neighbor_target)

    neighbor_source = set(list(graph.neighbors(source)))
    neighbor_target = set(list(graph.neighbors(target)))

    neighbor_sim.append(len(neighbor_source.intersection(neighbor_target)))
    # print neighbor_sim

    overlap_title.append(len(set(source_title).intersection(set(target_title))))
    temp_diff.append(int(source_info[1]) - int(target_info[1]))
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))


    if counter % 10000 == 0:
        print counter, "training examples processsed"
    counter += 1
# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
#training_features = np.array([overlap_title, temp_diff, comm_auth]).T
print 'add feautures'
training_features = np.array([overlap_title, temp_diff, comm_auth,tfidf_cos,neighbor_count_source,neighbor_count_target,neighbor_sim]).T

# scale
print 'scale'
training_features = preprocessing.scale(training_features)

# convert labels into integers then into column array
print 'labels'
labels = [int(element[2]) for element in training_set]
labels = list(labels)
labels_array = np.array(labels)

# print "evaluating"


# #evaluation
# kf = KFold(len(training_set), n_folds=10)
# sumf1=0
# for train_index, test_index in kf:
#     X_train, X_test = training_features[train_index], training_features[test_index]
#     y_train, y_test = labels_array[train_index], labels_array[test_index]
#     # initialize basic SVM
#     # classifier = svm.LinearSVC()
#     classifier = svm.SVC()
#     # train
#     classifier.fit(X_train, y_train)
#     pred=classifier.predict(X_test)
#     sumf1+=f1_score(pred,y_test)

# print "\n\n"
# print sumf1/10.0



# number of overlapping words in title
overlap_title_test = []

# temporal distance between the papers
temp_diff_test = []

# number of common authors
comm_auth_test = []

tfidf_cos_test=[]

neighbor_count_source_test=[]
neighbor_count_target_test=[]

#similarity based on neighbors
# neighbor_count_source.append(len(list(graph.neighbors(source))))
# neighbor_count_target.append(len(list(graph.neighbors(target))))

neighbor_sim_test=[]




counter = 0


for i in xrange(len(testing_set)):
    source = testing_set[i][0]
    target = testing_set[i][1]
    
    
    source_info_test = node_info[ID_pos[source]]
    target_info_test = node_info[ID_pos[target]]
    
	# convert to lowercase and tokenize
    source_title_test = source_info_test[2].lower().split(" ")
	# remove stopwords
    source_title_test = [token for token in source_title_test if token not in stpwds]
    source_title_test = [stemmer.stem(token) for token in source_title_test]
    
    target_title_test = target_info[2].lower().split(" ")
    target_title_test = [token for token in target_title_test if token not in stpwds]
    target_title_test = [stemmer.stem(token) for token in target_title_test]
    
    source_auth_test = source_info_test[3].split(",")
    target_auth_test = target_info_test[3].split(",")

    source_sim_test = M[ID_pos[source],:].toarray()[0]
    target_sim_test = M[ID_pos[target],:].toarray()[0]
    temp_cosine=0.0
    if np.linalg.norm(source_sim_test)!=0 and np.linalg.norm(target_sim_test)!=0:
        temp_cosine = cosine(source_sim_test,target_sim_test)
    
    tfidf_cos_test.append(temp_cosine)

    neighbor_source = len(list(graph.neighbors(source)))
    neighbor_target = len(list(graph.neighbors(target)))
    neighbor_count_source_test.append(neighbor_source)
    neighbor_count_target_test.append(neighbor_target)

    neighbor_source = set(list(graph.neighbors(source)))
    neighbor_target = set(list(graph.neighbors(target)))

    neighbor_sim_test.append(len(neighbor_source.intersection(neighbor_target)))
    
    overlap_title_test.append(len(set(source_title_test).intersection(set(target_title_test))))
    temp_diff_test.append(int(source_info_test[1]) - int(target_info_test[1]))
    comm_auth_test.append(len(set(source_auth_test).intersection(set(target_auth_test))))
   
    
    if counter % 10000 == 0:
        print counter, "testing examples processsed"
        counter += 1	
# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
test_features = np.array([overlap_title_test, temp_diff_test, comm_auth_test,tfidf_cos_test,
                            neighbor_count_source_test,neighbor_count_target_test,neighbor_sim_test]).T
test_features = preprocessing.scale(test_features)

classifier = svm.SVC()
classifier.fit(training_features,labels_array)
pred = classifier.predict(test_features)

with open('new_predictions_graph.csv','wb') as sub:
    csv_out=csv.writer(sub)
    for index,row in enumerate(pred):
        csv_out.writerow([index,row])
