# -*- coding: UTF-8 -*-
from lmnn import metricLearning
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import heapq
import numpy as np
import joblib
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Cluster():
    def __init__(self, l, net, dataset, method=1):
        self.l = l
        self.generateFeature('Datasets/'+dataset+net)
        # self.firstCluster(dataset+net)
        self.k = 3
        self.lmnn = metricLearning(self.k, self.feature, 100)
        self.dis_rank=[]
        self.n=len(np.load('Datasets/'+dataset+'labels.npy'))
        self.method=method
        print(self.n)

    def dist(self, vecA, vecB):
        if self.method==1:
            return np.sqrt(sum(np.power(vecA - vecB, 2)))
        if self.method==2:
            a_norm = np.linalg.norm(vecA)
            b_norm = np.linalg.norm(vecB)
            cos = np.dot(vecA,vecB)/(a_norm*b_norm)
            return cos
    
    def calcuProb(self, idx):
        # calculate cofidence
        return self.softmax([self.dist(self.lmnn.transformed_features[idx], i) for i in self.anchor])
    
    def softmax(self, x):
        """ softmax function """
        x=np.array(x) 
        x*=-1     
        x -= np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
        return x
    
    def calcuEntrophy(self, x):
        return sum([-i*np.log(i) if i!=0 else 0 for i in x])
    
    def generateFeature(self, path):
        print("path:"+path)
        self.feature=np.load(path+'_features.npy', allow_pickle=True)
    
    def calCenter(self):
        first=[0 for i in range(self.l)]
        second=[0 for i in range(self.l)]
        for idx,i in enumerate(self.confidence):
            for j in range(self.l):
                if i[j]>self.confidence[first[j]][j]:
                    first[j]=idx
                elif i[j]>self.confidence[second[j]][j]:
                    second[j]=idx
        # print(first)
        # print(second)
        return first, second

    def firstCluster(self, path):
        # time_start=time.time()
        # first = KMeans(n_clusters=self.l, random_state=0).fit(self.feature)
        # time_end=time.time()
        # print('clustering cost ',(time_end-time_start)//60,' min')
        # #joblib.dump(filename='inception_cluster.model',value=first)
        # #first=joblib.load('inception_cluster.model')
        # centers = first.cluster_centers_
        # confidence=[]
        # time_start=time.time()
        # for i in self.feature:
        #     confidence.append(self.softmax([self.dist(i,j) for j in centers]))
        # time_end=time.time()
        # np.save('Datasets/'+path+'_cluster_confidence.npy', confidence)
        # print('confidence cost ',(time_end-time_start)//60,' min')
        self.confidence = np.load('Datasets/'+path+'_cluster_confidence.npy')
        self.n = len(self.confidence)
        return self.calCenter()

    def restCluster_knn(self, reallabelDic, halflabel, reallabel):
        # train lmnn
        train_features=[]
        train_labels=[]
        q = len(reallabelDic)
        idxs=[]
        self.anchor = []
        for label in range(q):
            temp_index=[i[0] for i in reallabelDic[label]]
            temp_feature = self.feature[temp_index]
            m = len(temp_feature)
            train_labels.extend([label]*m)
            train_features.extend(temp_feature)
            idxs.extend(temp_index)
            self.anchor.append(np.sum(np.array(temp_feature),axis=0)/m)
        self.lmnn.trainLMNN(train_features, train_labels)
        self.anchor = self.lmnn.lmnn.transform(self.anchor)
        self.classification={i:[] for i in reallabelDic}
        knn = KNeighborsClassifier(n_neighbors=self.k, weights='distance', p=2)
        # print("testFeature length:"+str(len(self.lmnn.transformed_features[idxs][0])))
        knn.fit(self.lmnn.transformed_features[idxs],train_labels)
        idxs = []
        for i in range(self.n):
            if i in reallabel:
                continue
            idxs.append(i)
        temp = knn.predict_proba(self.lmnn.transformed_features[idxs])
        self.confidence = [[] for i in range(self.n)]
        for idx in range(self.n):
            if idx in reallabel:
                continue
            self.confidence[idx] = temp[idxs.index(idx)]
            if idx in halflabel:
                self.confidence[idx][halflabel[idx]]=0
                if np.sum(self.confidence[idx])!=0:
                    self.confidence[idx] = self.confidence[idx]/np.sum(self.confidence[idx])
                else:
                    continue
            model_label=np.argmax(self.confidence[idx])
            #heapq.heappush(self.classification[model_label],CompareAble(idx,self.confidence[idx][model_label]))
            self.classification[model_label].append((self.confidence[idx][model_label],idx))
        
    def restCluster(self, reallabelDic, halflabel, reallabel):
        train_features = []
        train_labels = []
        q = len(reallabelDic)
        self.anchor = []
        for label in range(q):
            temp_index=[i[0] for i in reallabelDic[label]]
            temp_feature = self.feature[temp_index]
            m = len(temp_feature)
            train_labels.extend([label]*m)
            train_features.extend(temp_feature)
            self.anchor.append(np.sum(np.array(temp_feature),axis=0)/m)
        self.lmnn.trainLMNN(train_features, train_labels)
        self.anchor = self.lmnn.lmnn.transform(self.anchor)
        self.max_class=[0 for i in range(self.n)]
        self.threshold=[0.6 for i in range(q)]
        self.amateur_annotate=[-1 for i in range(q)]
        self.confidence=[[] for i in range(self.n)]
        self.maxconfi=[]
        self.entropy=[]
        self.classification={i:[] for i in reallabelDic}
        for idx in range(self.n):
            if idx in reallabel:
                continue
            self.confidence[idx] = self.calcuProb(idx)
            if idx in halflabel:
                self.confidence[idx][halflabel[idx]]=0
                if np.sum(self.confidence[idx])!=0:
                    self.confidence[idx] = self.confidence[idx]/np.sum(self.confidence[idx])
                else:
                    self.max_class[idx]=self.l
                    continue
            model_label=np.argmax(self.confidence[idx])
            #heapq.heappush(self.classification[model_label],CompareAble(idx,self.confidence[idx][model_label]))
            self.maxconfi.append((self.confidence[idx][model_label],idx))
            self.entropy.append((self.calcuEntrophy(self.confidence[idx]),idx))
            self.classification[model_label].append((self.confidence[idx][model_label],idx))
            # self.max_class[idx]=model_label
            # origin=self.threshold[model_label]
            # cur=abs(self.confidence[idx][model_label]-0.5) # 业余者标离0.5最近的样本
            # if cur<origin:
            #     self.threshold[model_label]=self.confidence[idx][model_label]
            #     self.amateur_annotate[model_label]=idx
        self.confidence = np.array(self.confidence)

    def rankClass(self, reallabelDic):
        std_rank=[]
        num_rank=[]
        num_avg=0
        n=len(reallabelDic)
        for label in range(n):
            temp_index=[i[0] for i in reallabelDic[label]]
            temp_feature = self.lmnn.transformed_features[temp_index]
            std_rank.append(np.var([self.dist(feature, self.anchor[label]) for feature in temp_feature]))
            m=len(temp_feature)
            num_avg+=m
            num_rank.append(m)
        avg=np.mean(std_rank)
        std_rank/=avg
        num_rank=np.array(num_rank)/(num_avg/n)
        amateur_rank=np.array([std_rank[i]+num_rank[i] for i in range(n)])
        expert_rank=np.array([-std_rank[i]+num_rank[i] for i in range(n)])
        self.amateur_rank=amateur_rank.argsort()
        self.expert_rank=expert_rank.argsort()
        self.num_rank=num_rank.argsort()
    
    def distolabelRank(self, reallabel):
        self.lmnn.updateFarthest(reallabel)
        self.dis_rank=list(self.lmnn.distance_to_labeled.argsort()[::-1])
