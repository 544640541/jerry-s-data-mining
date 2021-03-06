import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

def InitCenter(k,m,x_train):
 #取数据集中前k个点作为初始中心
  Center = np.zeros([k,n])         #从样本中随机取k个点做初始聚类中心
  np.random.seed(15)            #设置随机数种子
  for i in range(k):
    x = np.random.randint(m)
    Center[i] = np.array(x_train.iloc[x])
  return Center

def getDistense(x_train, k, m, Center):
  Distence=[]
  for j in range(k):
    for i in range(m):
      x = np.array(x_train.iloc[i, :])
      a = x.T - Center[j]
      Dist = np.sqrt(np.sum(np.square(a)))     #计算距离公式
      Distence.append(Dist)
  Dis_array = np.array(Distence).reshape(k,m)
  return Dis_array

def getNewCenter(x_train,k,n, Dis_array):
#获取新的密度吸引中心点
  cen = []
  axisx ,axisy,axisz= [],[],[]
  cls = np.argmin(Dis_array, axis=0)
  for i in range(k):
    train_i=x_train.loc[cls == i]
    x,y,z = list(train_i.iloc[:,1]),list(train_i.iloc[:,2]),list(train_i.iloc[:,3])
    axisx.append(x)
    axisy.append(y)
    axisz.append(z)
    meanC = np.mean(train_i,axis=0)
    cen.append(meanC)
  newcent = np.array(cen).reshape(k,n)
  NewCent=np.nan_to_num(newcent)
  return NewCent,axisx,axisy,axisz

def KMcluster(x_train,k,n,m,threshold):
  global axis_x, axis_y
  center = InitCenter(k,m,x_train)
  initcenter = center
  centerChanged = True
  t=0
  while centerChanged:
    Dis_array = getDistense(x_train, k, m, center)
    center ,axis_x,axis_y,axis_z= getNewCenter(x_train,k,n,Dis_array)
    err = np.linalg.norm(initcenter[-k:] - center)
    t+=1
    print('err of Iteration '+str(t),'is',err)
    plt.figure(1)
    p1,p2,p3 = plt.scatter(axis_x[0], axis_y[0], c='c'),plt.scatter(axis_x[1], axis_y[1], c='m'),plt.scatter(axis_x[2], axis_y[2], c='y')
    plt.legend(handles=[p1, p2, p3], labels=['0', '1', '2'], loc='best')
    plt.show()
    if err < threshold:
      centerChanged = False
    else:
      initcenter = np.concatenate((initcenter, center), axis=0)
  return center, axis_x, axis_y,axis_z, initcenter

if __name__=="__main__":
  x=pd.read_csv("iris.csv")
  x_train=x.iloc[:,1:5]
  m,n = np.shape(x_train)
  k = 3
  threshold = 0.1
  km,ax,ay,az,ddd = KMcluster(x_train, k, n, m, threshold)
  print('最终的聚类中心为： ', km)

  plt.figure(2)
  plt.scatter(km[0,1],km[0,2],c = 'k',s = 200,marker='x')
  plt.scatter(km[1,1],km[1,2],c = 'k',s = 200,marker='x')
  plt.scatter(km[2,1],km[2,2],c = 'k',s = 200,marker='x')
  p1, p2, p3 = plt.scatter(axis_x[0], axis_y[0], c='c'), plt.scatter(axis_x[1], axis_y[1], c='m'), plt.scatter(axis_x[2], axis_y[2], c='y')
  plt.legend(handles=[p1, p2, p3], labels=['0', '1', '2'], loc='best')
  plt.title('2-D')
  plt.show()

  plt.figure(3)
  TreeD = plt.subplot(111, projection='3d')
  TreeD.scatter(ax[0],ay[0],az[0],c='c')
  TreeD.scatter(ax[1],ay[1],az[1],c='m')
  TreeD.scatter(ax[2],ay[2],az[2],c='y')
  TreeD.set_zlabel('Z')     # 坐标轴
  TreeD.set_ylabel('Y')
  TreeD.set_xlabel('X')
  TreeD.set_title('3-D')
  plt.show()

