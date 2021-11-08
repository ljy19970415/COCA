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

#     def getNeighbors(self, target):
#         _, indices = self.nbrs.kneighbors([self.transformed_features[target]])
#         return indices[0][1:]
    
#     def updateFarthest(self, reallabel):
#         for latest_labeled in reallabel:
#             for idx,i in enumerate(self.distance_to_labeled):
#                 if self.distance_to_labeled[idx]==0:
#                     continue
#                 new_distance = np.sqrt(np.sum((self.transformed_features[latest_labeled]-self.transformed_features[idx])**2))
#                 self.distance_to_labeled[idx]=min(i,new_distance)
    
    def trainLMNN(self, train_features, train_labels):
        # if os.path.exists('lmnn.model'):
        #     self.lmnn = joblib.load(filename = 'lmnn.model')
        
        # print("preparing model")
        # time_start=time.time()
        # train a new lmnn model
        # self.lmnn = LMNN(n_neighbors=self.k, max_iter=self.max_iter, n_components=None)
        self.lmnn.fit(train_features, train_labels)
        # time_end=time.time()
        # print('preparing model cost ',(time_end-time_start)//60,' min')
        self.transformed_features=self.lmnn.transform(self.features)
