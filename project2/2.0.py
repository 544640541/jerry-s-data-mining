import numpy as np
import csv
import pandas as pd

#读取数据
A=pd.read_csv('iris.csv')
A=np.array(A)
A=np.mat(A[:,0:4])
name=range(len(A))

#特征空间
K=np.mat(np.zeros((len(A),10)))
for i in range(0,len(A)):
    for j in range(0,4):
        K[i,j]=A[i,j]**2
    for m in range(0,3):
        for n in range(m+1,4):
            j=j+1
            K[i,j]=2*np.sqrt(A[i,m]*A[i,n])
print("齐次二次核矩阵：")
print(K)

#特征空间的成对点积
K1=np.mat(np.zeros((len(A),len(A))))
for i in range(0,len(A)):
    for j in range(i,len(A)):
        K1[i,j]=(np.dot(K[i],K[j].T))
        K1[j,i]=K1[i,j]
print("变换到特征空间：")
print(K1)


#标准化矩阵
E=np.mean(K1)    #计算均值E
S=np.std(K1)    #计算标准差S
Normal_K1=(K1-E)/S    #标准化
print("标准化后矩阵为：")
print(Normal_K1)

#中心化矩阵
rows=K1.shape[0]
cols=K1.shape[1]
centered_K1=np.mat(np.zeros((rows,cols)))
for i in range(0,cols):
    centered_K1[:,i]=K1[:,i]-np.mean(K1[:,i])
print("中心化后矩阵为：")
print(centered_K1)


#归一化矩阵
normalized_K1=np.mat(np.zeros((rows,cols)))
for i in range(0,len(K1)):
    temp=np.linalg.norm(K1[i])
    normalized_K1[i]=K1[i]/np.linalg.norm(K1[i])
print("归一化后矩阵为：")
print(normalized_K1)

