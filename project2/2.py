import numpy as np
import pandas as pd

#读取数据，并将文本数据转化为矩阵
f = open("iris.txt", "r")
List = f.readlines()

list0 = []
for i in range(len(List)):
  list1 = List[i].strip().split(",")
  list1.pop()    #去掉最后一列属性
  list0.append(list1)    #加入list0

a=np.array(list0)    #转化为np数组
a=a.astype(float)    #转换为浮点类型

#计算核矩阵K
K=np.mat(np.zeros((len(a),len(a))) )   #生成一个长宽为len（a）的空矩阵
for m in range(len(a)):
  for n in range(len(a)):
     K[m,n]=(np.dot(a[m],a[n].T))**2
     K[n,m]=K[m,n]

print("核矩阵K为：")
print(K)

#中心化矩阵
E=np.mean(K)    #计算K矩阵均值
Center=K-E
print("中心化矩阵：")
print(Center)

#标准化矩阵
S=np.std(K)
Normal=(K-E)/S
print("标准化矩阵：")
print(Normal)