# -*- coding: UTF-8 -*-
from pylmnn import LargeMarginNearestNeighbor as LMNN
# from metric_learn import LMNN
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import joblib
import numpy as np
import time
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class metricLearning():
    def __init__(self, k, features, max_iter):
        self.k=k
        self.features = features
        self.transformed_features = features.copy()
        self.max_iter=max_iter
        self.lmnn = LMNN(n_neighbors=self.k, max_iter=self.max_iter, n_components=None)
        self.nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='brute').fit(features)
        # 记录所有点距离已标注点集的距离，若该点已在标注点集中，则距离为0，否则为距离已标注点集中所有点距离的最小值
        self.distance_to_labeled=np.array([float("inf")]*len(features))

    def getNeighbors(self, target):
        # 获得k近邻id 
        #返回距离每个点k个最近的点和距离指数，indices可以理解为表示点的下标，distances为距离
        _, indices = self.nbrs.kneighbors([self.transformed_features[target]]) # 返回的indices为最近邻点在transformed_features中的下标(自己总在第一个)，在我们的情境里，下标即id
        return indices[0][1:]
    
    def updateFarthest(self, reallabel):
        # latest_labeled为最新被标注的点的id
        # 更新所有点到标注点集的距离
        for latest_labeled in reallabel:
            for idx,i in enumerate(self.distance_to_labeled):
                if self.distance_to_labeled[idx]==0:
                    continue
                new_distance = np.sqrt(np.sum((self.transformed_features[latest_labeled]-self.transformed_features[idx])**2))
                self.distance_to_labeled[idx]=min(i,new_distance)
    
    def trainLMNN(self, train_features, train_labels):
        # print("preparing model")
        time_start=time.time()
        # if os.path.exists('lmnn.model'):
        #     self.lmnn = joblib.load(filename = 'lmnn.model')
        self.lmnn.fit(train_features, train_labels)
        time_end=time.time()
        # print('preparing model cost ',(time_end-time_start)//60,' min')
        # joblib.dump(filename='middle_point/lmnn.model',value=self.lmnn)
        self.transformed_features=self.lmnn.transform(self.features)
        # algorithm:{‘auto’，‘ball_tree’，‘kd_tree’，‘brute’}, 可选
        # brute是蛮力搜索, 也就是线性扫描, 当训练集很大时, 计算非常耗时
        # kd_tree, 构造kd树存储数据以便对其进行快速检索的树形数据结构, kd树也就是数据结构中的二叉树. 以中值切分构造的树, 每个结点是一个超矩形, 在维数小于20时效率高
        # ball tree是为了克服kd树高纬失效而发明的, 其构造过程是以质心C和半径r分割样本空间, 每个节点是一个超球体.
        self.nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='brute').fit(self.transformed_features)
    
def random_discover_class():
    features = np.load('CUB_200/features.npy')
    labels = np.load('CUB_200/labels.npy')
    a=[i for i in range(len(labels))]
    count=200
    discoveredClass=set()
    while len(discoveredClass)!=200:
        ids=random.sample(a,count)
        discoveredClass=set(labels[ids])
        print("label_num:"+str(len(ids))+"; class_num:"+str(len(discoveredClass)))
        count+=100

def distance_to_label():
    features = np.load('CUB_200/features.npy')
    labels = np.load('CUB_200/labels.npy')

    k_train, k_test, n_components_1, max_iter = 3, 3, None, 150
    model = metricLearning(k_train, features, max_iter)
    #model.trainLMNN(train_features, train_labels)

    start=random.randint(0,199)
    discoveredClass=[labels[start]]
    model.updateFarthest(start)
    class_num=1
    label_num=1
    #f = open("distance_to_labeled.txt", "w")
    while len(discoveredClass)!=200:
        cur=np.argmax(model.distance_to_labeled)
        label_num+=1
        if labels[cur] not in discoveredClass:
            discoveredClass.append(labels[cur])
            class_num+=1
            print("class_num:"+str(class_num)+"; label_num:"+str(label_num))
            #print("class_num:"+str(class_num)+"; label_num:"+str(label_num), file=f)
        model.updateFarthest(cur)

if __name__=='__main__':
    # test_features = np.load('/raid/workspace/leijiayu/QATM-master/train_test_0.1/test_feature.npy')
    # test_labels = np.load('/raid/workspace/leijiayu/QATM-master/train_test_0.1/test_label.npy')
    # features = np.load('CUB_200/features.npy')
    # labels = np.load('CUB_200/labels.npy')

    # k_train, k_test, n_components_1, max_iter = 3, 3, None, 150
    # model = metricLearning(k_train, features, max_iter)
    # model.trainLMNN(train_features, train_labels)

    random_discover_class()