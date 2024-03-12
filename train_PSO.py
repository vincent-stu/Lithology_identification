import numpy as np
import matplotlib.pyplot as plt
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


def fit_fun(nest):  # 适应函数
    print("------------------------------------Start------------------------------------------")

    print("nest: ", nest)
    n_estimators, max_features, max_depth = int(nest[0, 0]), int(nest[0, 1]), int(nest[0, 2])
    print("type of nest: ", nest.dtype)
    print("n_estimators: ", n_estimators)
    print("max_features: ", max_features)
    print("max_depth: ", max_depth)
    svm_clf = ExtraTreesClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, random_state=42)
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


class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim):
        self.__pos = np.random.uniform(1, x_max, (1, dim))  # 粒子的位置
        self.__vel = np.random.uniform(-max_vel, max_vel, (1, dim))  # 粒子的速度
        self.__bestPos = np.zeros((1, dim))  # 粒子最好的位置
        self.__fitnessValue = fit_fun(self.__pos)  # 适应度函数值

    def set_pos(self, value):
        self.__pos = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, value):
        self.__bestPos = value

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, value):
        self.__vel = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class PSO:
    def __init__(self, dim, size, iter_num, x_max, max_vel, tol, best_fitness_value=0, C1=2, C2=2, W=1):
        self.C1 = C1
        self.C2 = C2
        self.W = W
        self.dim = dim  # 粒子的维度
        self.size = size  # 粒子个数
        self.iter_num = iter_num  # 迭代次数
        self.x_max = x_max
        self.max_vel = max_vel  # 粒子最大速度
        self.tol = tol  # 截至条件
        self.best_fitness_value = best_fitness_value
        self.best_position = np.zeros((1, dim))  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值

        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim) for i in range(self.size)]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, value):
        self.best_position = value

    def get_bestPosition(self):
        return self.best_position

    # 更新速度
    def update_vel(self, part):
        vel_value = self.W * part.get_vel() + self.C1 * np.random.rand() * (part.get_best_pos() - part.get_pos()) \
                    + self.C2 * np.random.rand() * (self.get_bestPosition() - part.get_pos())
        vel_value[vel_value > self.max_vel] = self.max_vel
        vel_value[vel_value < -self.max_vel] = -self.max_vel
        part.set_vel(vel_value)

    # 更新位置
    def update_pos(self, part):
        pos_value = part.get_pos() + part.get_vel()
        print("pos_val: ", pos_value)
        for i in range(3):
            if pos_value[0, i] <= 1:
                pos_value[0, i] = 1

        part.set_pos(pos_value)
        value = fit_fun(part.get_pos())
        if value > part.get_fitness_value():
            part.set_fitness_value(value)
            part.set_best_pos(pos_value)
        if value > self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            self.set_bestPosition(pos_value)

    def update_ndim(self):

        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
            self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
            print('第{}次最佳适应值为{}'.format(i, self.get_bestFitnessValue()))
            if self.get_bestFitnessValue() >= self.tol:
                break

        return self.fitness_val_list, self.get_bestPosition()

if __name__ == '__main__':
    # test
    pso = PSO(dim=3, size=150, iter_num=100, x_max=166, max_vel=10, tol=1, C1=2, C2=2, W=1)
    fit_var_list, best_pos = pso.update_ndim()
    print("最优位置:" + str(best_pos))
    print("最优解:" + str(fit_var_list[-1]))
    plt.plot(range(len(fit_var_list)), fit_var_list, alpha=0.5)






