# -*- coding: UTF-8 -*-
from lmnn import metricLearning
from sklearn.cluster import KMeans
import heapq
import numpy as np
import joblib
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Cluster():
    def __init__(self, l, net, dataset):
        self.l = l
        self.generateFeature(dataset+net)
        # self.firstCluster(dataset+net)
        self.lmnn = metricLearning(3, self.feature, 100)
        self.dis_rank=[]
        self.n=len(np.load(dataset+'labels.npy'))
        print(self.n)

    def distEclud(self, vecA, vecB):
        '''
        输入：向量A和B
        输出：A和B间的欧式距离
        '''
        return np.sqrt(sum(np.power(vecA - vecB, 2)))
    
    def calcuProb(self, idx):
        # 计算idx的图片在各类别上的置信度
        return self.softmax([self.distEclud(self.lmnn.transformed_features[idx], i) for i in self.anchor])
    
    def softmax(self, x):
        """ softmax function """
        x=np.array(x) 
        x*=-1     
        x -= np.max(x) #为了稳定地计算softmax概率， 一般会减掉最大的那个元素
        x = np.exp(x) / np.sum(np.exp(x))
        return x
    
    def calcuEntrophy(self, x):
        return sum([-i*np.log(i) if i!=0 else 0 for i in x])
    
    def generateFeature(self, path):
        # 提取所有图片的特征
        print("path:"+path)
        self.feature=np.load(path+'_features.npy')
    
    def calCenter(self):
        first=[0 for i in range(self.l)]
        second=[0 for i in range(self.l)]
        for idx,i in enumerate(self.confidence):
            for j in range(self.l):
                if i[j]>self.confidence[first[j]][j]:
                    first[j]=idx
                elif i[j]>self.confidence[second[j]][j]:
                    second[j]=idx
        return list(set(first)), list(set(second))

    def firstCluster(self, path):
        time_start=time.time()
        first = KMeans(n_clusters=self.l, random_state=0).fit(self.feature)
        time_end=time.time()
        print('clustering cost ',(time_end-time_start)//60,' min')
        #joblib.dump(filename='inception_cluster.model',value=first)
        #first=joblib.load('inception_cluster.model')
        centers = first.cluster_centers_
        confidence=[]
        time_start=time.time()
        for i in self.feature:
            confidence.append(self.softmax([self.distEclud(i,j) for j in centers]))
        time_end=time.time()
        np.save(dataset+net+'_cluster_confidence.npy', confidence)
        print('confidence cost ',(time_end-time_start)//60,' min')
        self.confidence = np.load(path+'_cluster_confidence.npy')
        self.n = len(self.confidence)
        return self.calCenter()

    def restCluster(self, reallabelDic, halflabel, reallabel, unlabel):
        train_features=[]
        train_labels=[]
        q=len(reallabelDic)
        # 设置各类别锚点
        self.anchor=[]
        # 初始化anchor
        for label in range(q):
            temp_feature = self.feature[reallabelDic[label]]
            m = len(temp_feature)
            train_labels.extend([label]*m)
            train_features.extend(temp_feature)
            self.anchor.append(np.sum(np.array(temp_feature),axis=0)/m)
        # 训练lmnn
        self.lmnn.trainLMNN(train_features, train_labels)
        # 更新anchor
        self.anchor = self.lmnn.lmnn.transform(self.anchor)
        # 设置记录
        self.max_class=[0 for i in range(self.n)] # 记录每个图片此时置信度最大的类别, max_class[i]为idx=i的图片置信度最大的类别的索引
        self.threshold=[0.6 for i in range(q)] # 记录每个类别的置信度门槛值
        self.amateur_annotate=[-1 for i in range(q)] #记录每个类别amateur应该标注的样本的id
        # 初始化confidence
        self.confidence=[[] for i in range(self.n)]
        # 记录模型现在对unlabel和halflabel的分类结果
        self.classification={i:[] for i in reallabelDic}
        # 记录所有非真标注样本的熵值
        self.entropy={i:[] for i in reallabelDic}
        # 计算数据距各类别锚点的距离，得出置信度
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
            self.classification[model_label].append((self.confidence[idx][model_label],idx))
            self.entropy[model_label].append((self.calcuEntrophy(self.confidence[idx]),idx))
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
            temp_feature = self.lmnn.transformed_features[reallabelDic[label]]
            std_rank.append(np.var([self.distEclud(feature, self.anchor[label]) for feature in temp_feature]))
            m=len(temp_feature)
            num_avg+=m
            num_rank.append(m)
        avg=np.mean(std_rank)
        std_rank/=avg
        num_rank=np.array(num_rank)/(num_avg/n)
        amateur_rank=np.array([std_rank[i]+num_rank[i] for i in range(n)])
        expert_rank=np.array([-std_rank[i]+num_rank[i] for i in range(n)])
        self.amateur_rank=amateur_rank.argsort() # 方差小，样本小在前
        self.expert_rank=expert_rank.argsort() # 方差大，样本小在前
        self.num_rank=num_rank.argsort() # 样本少的在前
    
    def distolabelRank(self, reallabel):
        self.lmnn.updateFarthest(reallabel)
        self.dis_rank=list(self.lmnn.distance_to_labeled.argsort()[::-1])

if __name__=='__main__':
    info=np.load('Stanford Dogs_records/resnet50_4/8/info.npy')
    nums=len(np.load('Stanford Dogs/labels.npy'))
    print(info)
    print((info[1]+info[3])/nums)
    print(nums)

    info=np.load('Stanford Cars_records/resnet50_4/9/info.npy')
    nums=len(np.load('Stanford Cars/labels.npy'))
    print(info)
    print((info[1]+info[3])/nums)
    print(nums)
    # nets=['vgg','mobilenet','resnet']
    # dataset='Stanford Dogs/'
    # for net in nets:
    #     print(dataset+net)
    #     clus=Cluster(120, net, dataset)
    #     clus.firstCluster(dataset+net)
    # labels=np.load('Stanford Dogs/labels.npy')
    # paths=np.load('Stanford Dogs/paths.npy')
    # print(len(labels))
    # print(len(paths))
    # labels=np.load('Stanford Cars/labels.npy')
    # paths=np.load('Stanford Cars/paths.npy')
    # print(len(labels))
    # print(len(paths))
