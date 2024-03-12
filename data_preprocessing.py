import os
import numpy as np

files_path = r'./data_gather/seri'
lst = np.zeros((166, 2), dtype='float')

for filename in os.listdir(files_path):
    lujing = os.path.join(files_path, filename)
    with open(lujing, "r") as f:
        lines = f.readlines()

    with open(lujing, 'w') as f:
          f.write(''.join(lines[3:]))
print("已完成删除前三行无效信息！")

for filename in os.listdir(files_path):
    lujing = os.path.join(files_path, filename)
    data = np.loadtxt(lujing)
    for line in range(166):
        lst[line][0] = data[line][0]
        lst[line][1] = data[line][1]

    np.savetxt(lujing, lst, delimiter="  \t", newline="\n")

print("已完成去除每行前面的空格！")


