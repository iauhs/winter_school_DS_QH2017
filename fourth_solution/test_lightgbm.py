import lightgbm as lgb
import pandas as pd
import numpy as np
from numpy import genfromtxt



training_feautres = np.genfromtxt("../output/training_feautres.csv")
training_labels = np.genfromtxt("../output/training_labels.csv")
test_features = np.genfromtxt("../output/test_features.csv")
# my_data = genfromtxt('my_file.csv', delimiter=',')

train_data = lgb.Dataset(training_feautres, label=training_labels)

test_data = train_data.create_valid(test_features)


param = {'num_leaves':31, 'num_trees':100, 'objective':'binary' }
param['metric'] = 'auc'


num_round = 10
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data] )


num_round = 10
lgb.cv(param, train_data, num_round, nfold=5)

ypred = bst.predict(train_data)

np.savetxt('../output/submission_lightgbm.csv', numpy.c_[range(1, len(testData) + 1), preds], delimiter=',',
              header='id,prediction', comments='', fmt='%d')
