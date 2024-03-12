import numpy as np      #导入numpy包
import os               #导入os包


def file_name(file_dir):             #获取文件夹下的文件名
    for root,dirs, files in os.walk(file_dir):
        print(root)
        print(files)
    return files


file=file_name('./data_gather/other')   #调用文件夹下的文件
number = len(file)
list=np.zeros((number+1,167),dtype='float')
for q in range(number):
    path = "./data gather/other/" + file[q]
    data = np.loadtxt(path)
    trans = np.transpose(data)
    for info in range(166):
        list[0][info + 1] = trans[0][info]
    for info in range(166):
        list[q + 1][info + 1] = trans[1][info]
        list[q + 1][0] = q + 1
list = np.round(list,decimals=6)
np.savetxt('./data gather/other/other.csv',list,delimiter=',')   #将list写入文件
print('end')