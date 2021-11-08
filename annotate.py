# -*- coding: UTF-8 -*-
import numpy as np
import os
import math
import random
from cluster import Cluster

class Annotate():
    def __init__(self, dataset, e, a, rootpath, net):
        self.e = e
        self.a = a
        self.q = e//a 
        # self.classReward = 0 
        # self.classTruth={}
        self.rootpath=rootpath
        self.dataset=dataset
        self.readDataset()
        self.B = self.n*self.e
        self.left = self.B
        self.annotation=[self.l]*self.n
        self.unlabel=[i for i in range(self.n)]
        self.reallabel = []
        self.halflabel = {}
        self.discoveredClass = []
        self.error = 1
        self.correct = 0.5
        self.e_num=0
        self.a_num=0
        self.a_correct=0
        self.a_fp=0
        self.a_tp=0 
        self.a_fn=0 
        self.a_tn=0 
        
        self.lastAccuracy=0
        self.myCluster=Cluster(self.l, net, dataset)
        self.reallabelDic={}

        self.prev=len(self.discoveredClass)
        self.no_progress=0
        self.tail_flag=False

        self.max_ae_ratio = 4
        self.min_ae_ratio = 1
        self.ae_ratio=self.min_ae_ratio
        self.batch_size= 64
    
    def genAccuracy(self):
        mu,sigma=0.9,0.03
        s=np.random.normal(mu,sigma,1)
        return s

    def readDataset(self):
        self.label = np.load('Datasets/'+self.dataset+'labels.npy')
        self.n = len(self.label)
        self.l = len(set(self.label))
        print("total catagories: "+str(self.l))
    
    def continue_from_middle(self):
        print("begin")
        self.reallabel=list(np.load(self.rootpath+'middle_point/reallabel.npy'))
        self.halflabel=np.load(self.rootpath+'middle_point/halflabel.npy', allow_pickle=True).item()
        self.unlabel=list(np.load(self.rootpath+'middle_point/unlabel.npy'))
        self.discoveredClass=list(np.load(self.rootpath+'middle_point/discoveredClass.npy'))
        self.annotation=list(np.load(self.rootpath+'middle_point/annotation.npy'))
        self.reallabelDic=np.load(self.rootpath+'middle_point/reallabelDic.npy', allow_pickle=True).item()
        # self.myCluster.confidence=np.load(rootpath+'middle_point/confidence.npy', allow_pickle=True)
        ae=np.load(self.rootpath+'middle_point/info.npy')
        self.left, self.e_num, self.a_num, self.a_correct, self.lastAccuracy=int(ae[0]),int(ae[1]),int(ae[2]),int(ae[3]),ae[4]
        print("end")
    
    def saveMiddle(self, k):
        # np.save(rootpath+'middle_point/confidence.npy',self.myCluster.confidence)
        np.save(self.rootpath+'middle_point/reallabel.npy', self.reallabel)
        np.save(self.rootpath+'middle_point/halflabel.npy', self.halflabel)
        np.save(self.rootpath+'middle_point/unlabel.npy', self.unlabel)
        np.save(self.rootpath+'middle_point/annotation.npy', self.annotation)
        np.save(self.rootpath+'middle_point/info.npy', [self.left, self.e_num, self.a_num, self.a_correct, self.lastAccuracy])
        np.save(self.rootpath+'middle_point/discoveredClass.npy', self.discoveredClass)
        np.save(self.rootpath+'middle_point/reallabelDic.npy', self.reallabelDic)
        np.save(self.rootpath+'middle_point/k.npy',[k])

    def annotate(self,k, isContinue, eSelect, aSelect, net):
        if isContinue:
            self.continue_from_middle()
        if k==1: 
            self.firstBatch(self.dataset+net)
        else:
            self.restBatch(eSelect, aSelect)
        if self.left<=0:
            return False
        if len(self.reallabel)==self.n:
            return False
        self.saveMiddle(k)
        return True
    
    def firstBatch(self, path):
        first, second = self.myCluster.firstCluster(path)
        self.expertAnnotate(first)
        correct=0
        temp=set()
        for clas,idx in enumerate(second):
            if idx in first or idx in temp:
                continue
            else:
                temp.add(idx)
            self.a_num += 1
            self.left -= self.a
            isCorrect=1
            if self.label[idx]==self.label[first[clas]]:
                self.a_correct += 1
                self.a_tp+=1
                correct+=1
                self.reallabelDic[self.annotation[first[clas]]].append((idx,isCorrect,1))
                self.reallabel.append(idx)
                self.annotation[idx] = self.annotation[first[clas]]
            else:
                self.a_tn+=1
                self.halflabel[idx]=[self.annotation[first[clas]]]
            self.unlabel.remove(idx)
        self.prev=len(self.discoveredClass)
        self.no_progress=0
        self.tail_flag=False
        self.lastAccuracy=correct/(len(second))

    def restBatch(self, eSelect, aSelect):
        self.updateModel()
        a_annotate=self.genAmateurAnnotate(aSelect)
        self.amateurAnnotate(a_annotate)
        e_annotate=self.genExpertAnnotate(a_annotate, eSelect)
        self.expertAnnotate(e_annotate)
        if len(self.discoveredClass)<self.l:
            self.isTail()
            print('no progress:'+str(self.no_progress)+'; isTail:'+str(self.tail_flag))
    
    def genAmateurAnnotate(self, aSelect):
        a_idlist=[]
        # num=int(np.ceil(self.batch_size*self.lastAccuracy)) if int(np.ceil(self.batch_size*self.lastAccuracy))!=0 else (10 if self.n-len(self.reallabel)>10 else 0)
        #print("expert_num:"+str(n2)+"; punish:"+str(n1)+"; ae_ratio:"+str(self.ae_ratio)+"; anum:"+str(n3))
        num=min(np.ceil((self.batch_size-int(np.ceil(self.batch_size*self.lastAccuracy)))*self.ae_ratio),len(self.unlabel)+len(self.halflabel))
        if num==0:
            return a_idlist
        #num=min(num,len(self.myCluster.amateur_rank))
        if aSelect==1:
            # random
            legal=list(self.halflabel.keys())+self.unlabel
            return random.sample(legal,num) if num<len(legal) else legal
        if aSelect==2:
            # min entropy
            self.myCluster.entropy.sort(key=lambda x:-x[0])
            start=max(0,len(self.myCluster.entropy)-num)
            return [i[1] for i in self.myCluster.entropy[start:]]
        if aSelect==3:
            # max confidence
            self.myCluster.maxconfi.sort(key=lambda x:-x[0])
            end=min(len(self.myCluster.maxconfi)-1,num)
            return [i[1] for i in self.myCluster.maxconfi[:end]]
        count=0
        rank_margin = num % len(self.myCluster.amateur_rank)
        if aSelect==4: 
            # COCA
            class_num=0
            for idx,label in enumerate(self.myCluster.amateur_rank):
                if len(self.myCluster.classification[label]):
                    class_num+=1
                    if idx<rank_margin:
                        limit=num//len(self.myCluster.amateur_rank) + 1
                    else:
                        limit=num//len(self.myCluster.amateur_rank)
                    self.myCluster.classification[label] = sorted(self.myCluster.classification[label], key=lambda x:abs(x[0]-0.5))
                    class_count=0
                    for item in self.myCluster.classification[label]:
                        a_idlist.append(item[1])
                        class_count+=1
                        count+=1
                        if class_count==limit:
                            break
                        if count==num:
                            break
                if count==num:
                    break
            if count<num:
                candidate=[]
                for i in range(self.n):
                    if i in self.reallabel or i in a_idlist:
                        continue
                    candidate.append(i)
                legal2=random.sample(candidate, num-count) if len(candidate)>num-count else candidate
                return a_idlist+legal2
        print("class_num:"+str(class_num))
        return a_idlist
    
    def isTail(self):
        if len(self.discoveredClass)==self.prev:
            self.no_progress+=1
        else:
            self.no_progress=0
        self.prev=len(self.discoveredClass)
        if self.no_progress>10:
            self.tail_flag=True
        else:
            self.tail_flag=False

    def genExpertAnnotate(self, a_annotate, eSelect):
        legal=[]
        print("lastAccuracy:"+str(self.lastAccuracy))
        print("batchsize:"+str(self.batch_size))
        num=self.batch_size-int(np.ceil(self.batch_size*self.lastAccuracy))
        print("expert_num:"+str(num))
        # num=int(self.batch_size-len(a_annotate))
        # random
        if eSelect==1 or (self.tail_flag and len(self.discoveredClass)<self.l):
            print("6 or class discover random select")
            # legal=list(set(list(self.halflabel.keys())+self.unlabel)-set(a_annotate))
            legal=list(self.halflabel.keys())+self.unlabel
            return random.sample(legal, num) if num<len(legal) else legal
        # all categories discovered
        if len(self.discoveredClass)==self.l: # 若类别发现结束
            if eSelect==2: # max entropy
                end=min(len(self.myCluster.entropy)-1,num)
                return [i[1] for i in self.myCluster.entropy[:end]]
            if eSelect==3: # least confidence
                start=max(0,len(self.myCluster.maxconfi)-num)
                return [i[1] for i in self.myCluster.maxconfi[start:]]
            if eSelect==4: # COCA
                print("done discover, expert select")
                count=0
                for label in self.myCluster.expert_rank:
                    if len(self.myCluster.classification[label]):
                        # threshold=self.myCluster.classification[label][0][0]
                        for item in self.myCluster.classification[label]:
                            #if item[0]<=threshold and item[1] not in a_annotate:
                            #if item[1] not in a_annotate:
                            if item[1] not in self.reallabel:
                                legal.append(item[1])
                                count+=1
                                break
                    if count==num:
                        break
                if count<num:
                    candidate=[]
                    for i in range(self.n):
                        if i in self.reallabel or i in legal:
                            continue
                        candidate.append(i)
                    legal2=random.sample(candidate, num-count) if len(candidate)>num-count else candidate
                    return legal+legal2
                return legal
        # category discovery
        print("class_discover, filter random select")
        for i in range(self.n):
            #if i in self.reallabel or i in a_annotate:
            if i in self.reallabel:
                continue
            # max_class=self.myCluster.max_class[i]
            max_class=np.argmax(self.myCluster.confidence[i])
            #在genAmateurAnnotate时已经对self.myCluster.classification排过序，按与0.5的距离进行升序排列
            if self.myCluster.confidence[i][max_class]<=self.myCluster.classification[max_class][0][0]:
                legal.append(i)
        return random.sample(legal, num) if num<len(legal) else legal

    def updateModel(self):
        self.myCluster.restCluster(self.reallabelDic, self.halflabel, self.reallabel)
        self.myCluster.rankClass(self.reallabelDic)

    def amateurAnnotate(self, idlist):
        correct=0
        res=[]
        class_correct=0
        total=0
        if idlist:
            pre_label= np.argmax(self.myCluster.confidence[idlist[0]])
        for idx in idlist:
            self.left -= self.a
            self.a_num += 1
            i = np.argmax(self.myCluster.confidence[idx])
            if i==pre_label:
                total+=1
            else:
                pre_label=i
                res.append(class_correct/total)
                class_correct=0
                total=1
            if self.label[idx]==self.discoveredClass[i]:
                self.a_correct += 1
                correct += 1
                self.reallabel.append(idx)
                self.reallabelDic[i].append((idx,1,1))
                self.annotation[idx]=i
                if i==pre_label:
                    class_correct+=1
                if idx in self.halflabel:
                    del self.halflabel[idx]
                else:
                    self.unlabel.remove(idx)
            else:
                if idx in self.halflabel:
                    self.halflabel[idx].append(i)
                else:
                    self.halflabel[idx]=[i]
                    self.unlabel.remove(idx)
        now_Accuracy=correct/len(idlist) if len(idlist)!=0 else 0
        if self.lastAccuracy<=now_Accuracy:
            self.ae_ratio=min(self.max_ae_ratio,self.ae_ratio+1)
        else:
            self.ae_ratio=max(self.min_ae_ratio,self.ae_ratio-1)
        self.lastAccuracy= now_Accuracy

    def expertAnnotate(self, idlist):
        temp=set()
        for idx in idlist:
            if idx in temp or idx in self.reallabel:
                continue
            else:
                temp.add(idx)
            self.left -= self.e
            self.e_num += 1
            if len(self.discoveredClass)<self.l and self.label[idx] not in self.discoveredClass:
                self.discoveredClass.append(self.label[idx])
                self.reallabelDic[len(self.discoveredClass)-1]=[(idx,1,0)]
                self.reallabel.append(idx)
            else:
                i = self.discoveredClass.index(self.label[idx])
                self.reallabelDic[i].append((idx,1,0))
                self.reallabel.append(idx)
            if idx in self.halflabel:
                del self.halflabel[idx]
            else:
                self.unlabel.remove(idx)
            self.annotation[idx] = self.discoveredClass.index(self.label[idx]) 
        
    def calcuPrecision(self):
        a = 0
        found_class_num = 0
        found_class_correct = 0
        for i in range(self.n):
            if self.label[i] in self.discoveredClass:
                found_class_num+=1
                if self.label[i] == self.annotation[i]:
                    found_class_correct+=1
            if self.label[i] == self.annotation[i]:
                a += 1
        return a/self.n, found_class_correct/found_class_num

    def isClassEven(self):
        a=[]
        for i in self.reallabelDic:
            a.append(len(self.reallabelDic[i]))
        a=np.array(a)
        return min(a), max(a), np.std(a, ddof = 1)
    
    def calBatchPrecision(self):
        for idx in self.unlabel:
            # print(len(self.myCluster.confidence[idx]))
            reference=self.myCluster.confidence[idx][:len(self.discoveredClass)]
            self.annotation[idx]=np.argmax(reference)
        for idx in self.halflabel:
            # 将confidence中index在self.halflabel[id]中的位置置0
            self.myCluster.confidence[idx][self.halflabel[idx]]=0
            # 再用argmax(confidence)算出此时的最大置信度指向的类别
            reference=self.myCluster.confidence[idx][:len(self.discoveredClass)]
            self.annotation[idx]=np.argmax(reference)  
        temp=[self.discoveredClass[i] for i in self.annotation]
        a=0
        for i in range(self.n):
            if self.label[i] == temp[i]:
                a += 1
        min_num, max_num, std = self.isClassEven()
        return a/self.n, self.B-self.left, len(self.discoveredClass), self.e_num, self.a_num, self.a_correct, min_num, max_num, std
