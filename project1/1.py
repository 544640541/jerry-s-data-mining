import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#读取数据
f = open("1.txt", "r")
rows = f.readlines()

#初始数据处理
list = []
for i in range(len(rows)):
    column_list = rows[i].strip().split(",")    #每一行以“，”为分隔符
    column_list.pop()    #去掉最后一列的分类属性
    list.append(column_list)    #加入list_source
a=np.array(list)    #转化为np数组
a=a.astype(float)    #转换为浮点类型

MeanVector=np.mean(a,axis=0)    #均值向量
print("向量均值为：")
print(MeanVector)

center=a-MeanVector    #中心化
innerProduct=np.dot(center.T,center)
print("内积为：")
print(innerProduct/len(center))    #打印计算出的内积

Kroneckerproduct=0
for i in range(len(center)):
    Kroneckerproduct = Kroneckerproduct+center[i].reshape(len(center[0]),1)*center[i]
print("外积为：")
print(Kroneckerproduct/len(center))    #打印计算出的外积

#求方差
list=[]
for i in range(len(a[0])):
    list.append(np.var(a.T[i]))
print("各列的方差：")
print(list)
maxIndex=list.index(max(list))
minIndex=list.index(min(list))
print("方差最大的属性所在列数为：",end=" ")
print(maxIndex+1)
print("方差最小的属性所在列数为：",end=" ")
print(minIndex+1)

#求矩阵协方差
Cov={}
for i in range(9):
    for j in range(i+1,10):
        st=str(i+1)+'-'+str(j+1)
        Cov[st]= np.cov(a.T[i],a.T[j])[0][1]    #遍历求协方差
print("协方差矩阵为：")
print(Cov)    #打印协方差矩阵
print("哪对属性的协方差最大：",end=" ")
print(max(Cov, key=Cov.get))    #取最大值打印
print("哪对属性的协方差最小：",end=" ")
print(min(Cov, key=Cov.get))    #取最小值打印

t=center.T    #通过中心化后的向量计算属性1和2的夹角
cor=np.corrcoef(t[0],t[1])    #计算第一列属性和第二列属性相关性
print("相关系数为：")
print(cor[0][1])    #打印相关系数

picture = plt.figure()
ax1 = picture.add_subplot(111)    #设置标题
ax1.set_title("Correlation scatter plots")
plt.scatter(t[0],t[1],color='c',marker='o')
plt.xlabel('Attributes 1')    #设置X轴标签
plt.ylabel('Attributes 2')    #设置Y轴标签
plt.show()

#def normfun(x, E, S):
#    pdf = np.exp(-((x - E) ** 2) / (2 * S ** 2)) / (S * np.sqrt(2 * np.pi))
#    return pdf

#计算E和s
E=np.mean(a,axis=0)[0]    #计算第一列均值
S=np.var(a.T[0])    #计算第一列标准差
fig = plt.figure()
ax1 = picture.add_subplot(111)
ax1.set_title("Probability density function")
# 绘制正态分布概率密度函数
x = np.linspace(E - 3 * S, E + 3 * S, 50)
y_sig = np.exp(-(x - E) ** 2 / (2 * S ** 2)) / (math.sqrt(2 * math.pi) * S)
plt.plot(x, y_sig, "k", linewidth=2)
plt.vlines(E, 0, np.exp(-(E - E) ** 2 / (2 * S ** 2)) / (math.sqrt(2 * math.pi) * S), colors="red",
           linestyles="dashed")
plt.vlines(E + S, 0, np.exp(-(E + S - E) ** 2 / (2 * S ** 2)) / (math.sqrt(2 * math.pi) * S),
           colors="m", linestyles="dotted")
plt.vlines(E - S, 0, np.exp(-(E - S - E) ** 2 / (2 * S ** 2)) / (math.sqrt(2 * math.pi) * S),
           colors="m", linestyles="dotted")
plt.xticks([E - S, E, E + S], ['μ-σ', 'μ', 'μ+σ'])
plt.xlabel('Attributes 1')
plt.ylabel('Attributes 2')
plt.title('Normal Distribution: $\mu = %.2f, $σ=%.2f' % (E, S))
plt.grid(True)
plt.show()
