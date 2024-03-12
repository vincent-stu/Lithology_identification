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
# 训练并导出模型
data = pd.read_csv('./data_gather/spectral_data.csv')
SavePath = './model/RF_model.pickle'
data = data.to_numpy()
data_Ysize = data.shape[1]
# data = np.loadtxt(file, delimiter=',')
X = data[:, 1: -1]
y = data[:,-1]
train_data, test_data, train_label, test_label = train_test_split(X, y, random_state=0, train_size=0.6,test_size=0.4,shuffle=True)
# svm_clf = SVC(kernel='linear', C=float('inf'), decision_function_shape='ovr')
#svm_clf = RandomForestClassifier(random_state=0, oob_score=True)
svm_clf = ExtraTreesClassifier(n_estimators=100, max_depth=2, random_state=42)
# svm_clf = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500)
# svm_clf = KNeighborsClassifier()
print('1')

# svm_clf = GaussianNB()
# svm_clf = MLPClassifier(
#                                     hidden_layer_sizes=(152,152),
#                                     activation='relu',
#                                     solver='adam',
#                                     alpha=1e-3)
# svm_clf = DecisionTreeClassifier(criterion='entropy')
svm_clf.fit(train_data, train_label.ravel())
file = open(SavePath, 'wb')
pickle.dump(svm_clf, file)
file.close()
train_score = svm_clf.score(train_data,train_label)
print("训练集：", train_score)
test_score = svm_clf.score(test_data,test_label)
print("测试集：", test_score)
y_hat = svm_clf.predict(test_data)
print(classification_report(test_label,y_hat))
# print('train_decision_function:\n',svm_clf.decision_function(train_data))#（90，3）
#训练集和测试集的预测结果
# trainPredict = (svm_clf.predict(train_data).reshape(-1, 1))
# testPredict = svm_clf.predict(test_data).reshape(-1, 1)
kappa = cohen_kappa_score(test_label,y_hat)
print(kappa)
# print(svm_clf.oob_score_)
print('*********************************************************************************************')












#将预测结果进行展示,首先画出预测点，再画出分类界面
#画图例和点集
# x1_min, x1_max = X[:, 0].min(), X[:, 0].max()   #x轴范围
# x2_min, x2_max = X[:, 0].min(), X[:, 1].max()   #y轴范围
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']   #指定默认字体
# cm_dark=matplotlib.colors.ListedColormap(['g', 'r', 'b'])  #设置点集颜色格式
# cm_light=matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])  #设置边界颜色
# plt.xlabel('花萼长度', fontsize=13)        #x轴标注
# plt.ylabel('花萼宽度', fontsize=13)        #y轴标注
# plt.xlim(x1_min, x1_max)                   #x轴范围
# plt.ylim(x2_min, x2_max)                   #y轴范围
# plt.title('鸢尾花SVM二特征分类')          #标题
#
# plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], s=30, cmap=cm_dark)  #画出测试点
# plt.scatter(test_data[1:10, 0], test_data[1:10, 1], c=test_label[:, 0], s=30, edgecolors='k', zorder=2, cmap=cm_dark) #画出预测点，并将预测点圈出
# #画分类界面
# x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]#生成网络采样点
# grid_test = np.stack((x1.flat, x2.flat), axis=1)#测试点
# grid_hat = svm_clf.predict(grid_test)# 预测分类值
# grid_hat = grid_hat.reshape(x1.shape)# 使之与输入的形状相同
# plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)# 预测值的显示
# plt.show()
###
# 进行图像分类
def readTif(filename):
    dataset = gdal.Open(filename)
    if dataset == None:
        print(filename + '文件无法打开')
    return dataset


def ReadPoints():
    water = readTif('./ZY1F_AHSI_E97.00_N42.36_20230615_007690_L1A0000461109_BSQ_radiacne_flaash_H1B.tiff')
    Tif_width = water.RasterXSize
    Tif_height = water.RasterYSize
    img_data = water.ReadAsArray(0, 0, Tif_width, Tif_height)


def writeTif(PreData, img_geotrans, img_proj, path):
    # if 'int8' in PreData.dtype.name:
    #     datatype = gdal.GDT_Byte
    # elif 'int16' in PreData.dtype.name:
    #     datatype = gdal.GDT_UInt16
    # else:
    #     datatype = gdal.GDT_Float32
    # if len(PreData.shape) == 3:
    #     img_band, img_height, img_width = PreData.shape
    # elif len(PreData.shape) == 2:
    #     temp = np.array([PreData])
    #     img_band, img_height, img_width = temp.shape
    # driver = gdal.GetDriverByName('GTiff')
    # dataset = driver.Create(path, int(img_width), int(img_height), int(img_band), datatype)
    # if (dataset != None):
    #     dataset.SetGeoTransform(img_geotrans)
    #     dataset.SetProjection(img_proj)
    # for i in range(img_band):
    #     dataset.GetRasterBand(i + 1).WriteArray(PreData[i])
    #     del dataset
    temp = np.array([PreData])
    img_band, img_height, img_width = temp.shape
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(path, int(img_width), int(img_height), int(img_band), gdal.GDT_Byte)
    if (dataset != None):
        dataset.SetGeoTransform(img_geotrans)
        dataset.SetProjection(img_proj)
    # for i in range(img_band):
    #     dataset.GetRasterBand(i + 1).WriteArray(PreData[i])
    #     del dataset
    dataset.GetRasterBand(1).WriteArray(PreData)
    del dataset
RFpath = './model/RF_model.pickle'
img_Path = './ZY1F_AHSI_E97.00_N42.36_20230615_007690_L1A0000461109_BSQ_radiacne_flaash_H1B.tiff'
Save_Path = "./ZY1F_RF_yanxing.tif"
dataset = readTif(img_Path)
Tif_width = dataset.RasterXSize
Tif_height = dataset.RasterYSize
Tif_geotrans = dataset.GetGeoTransform()
Tif_proj = dataset.GetProjection()
img_data = dataset.ReadAsArray(0, 0, Tif_width, Tif_height)
#img_data = img_data[10:20, :, :]

file = open(RFpath, 'rb')
rf_model = pickle.load(file)
file.close()

New_data = np.zeros((img_data.shape[0], img_data.shape[1] * img_data.shape[2]))
for i in range(img_data.shape[0]):
    New_data[i] = img_data[i].flatten()
New_data = New_data.swapaxes(0, 1)
print(New_data.shape[0])
print(New_data.shape[1])
lsw = rf_model.predict(New_data)
lsw = lsw.reshape(img_data.shape[1], img_data.shape[2])
lsw = lsw.astype(np.uint8)
writeTif(lsw, Tif_geotrans, Tif_proj, Save_Path)
print('end')