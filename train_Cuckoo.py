import pickle
from osgeo import gdal
from sklearn.svm import SVC
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from Cuckoo_search import cuckoo_search


data = pd.read_csv('./data_gather/spectral_data.csv')
SavePath = './model/RF_model.pickle'
data = data.to_numpy()
data_Ysize = data.shape[1]
# data = np.loadtxt(file, delimiter=',')
X = data[:, 1: -1]
y = data[:,-1]
train_data, test_data, train_label, test_label = train_test_split(X, y, random_state=0, train_size=0.6,test_size=0.4,shuffle=True)

def fit_func(nest):
    print("------------------------------------Start------------------------------------------")
    n_estimators, max_features, max_depth = map(int, nest)
    print("type of nest: ", nest.dtype)
    print("n_estimators: ", n_estimators)
    print("max_features: ", max_features)
    print("max_depth: ", max_depth)
    svm_clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth,max_features=max_features, random_state=42)
    svm_clf.fit(train_data, train_label.ravel())
    file = open(SavePath, 'wb')
    pickle.dump(svm_clf, file)
    file.close()
    y_hat = svm_clf.predict(test_data)
    kappa = cohen_kappa_score(test_label, y_hat)
    print("kappa: ", kappa)
    print("--------------------------------------end---------------------------------------------")
    print("\n")
    return kappa

best_nest, best_fitness = cuckoo_search(25, 3, fit_func, [50, 1, 1], [200,166,20], step_size=0.1)
print("best_nest: ", best_nest)

print('最大值为:%.5f, 在(%.5f, %.5f, %.5f)处取到!' % (best_fitness, best_nest[0], best_nest[1], best_nest[2]))