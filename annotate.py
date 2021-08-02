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
        self.q = e//a # 计算utility时表示业余者可以标注的次数
        # self.classReward = 0 # 具体大小取决于|\delta H|
        # self.classTruth={} # 记录字符串标签的id
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
        self.correct = 0.1
        self.batch_size=64
        self.e_num=0 # 专家标记的个数
        self.a_num=0 # 业余者标记的个数
        self.a_correct=0 # 业余者指出正确类别的次数
        self.a_fp=0 # 将错误的标签贴到样本上
        self.a_tp=0 # 将正确的标签贴到样本上
        self.a_fn=0 #将正确的样本排除
        self.a_tn=0 #将错误的样本排除
        
        self.lastAccuracy=0 #上一次的业余者识别精度
        self.myCluster=Cluster(self.l, net, dataset)
        self.reallabelDic={} # 记录每个类别下有哪些已标注样本

        self.prev=len(self.discoveredClass)
        self.no_progress=0
        self.tail_flag=False

        self.ae_ratio=15
    
    def readDataset(self):
        # 设置self.classTruth
        # with open(os.path.join('/datapool/workspace/leijiayu/MAAS/'+dataset+'/classes.txt'), 'r') as f:
        #     while True:
        #         line = f.readline()
        #         if not line:
        #             break
        #         class_id, class_name = [i for i in line.split()]
        #         self.classTruth[int(class_id)-1] = class_name
        # 设置self.label
        self.label = np.load(self.dataset+'labels.npy')
        # 设置self.n
        self.n = len(self.label)
        self.l = len(set(self.label))
    
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

    def annotate(self,k, isContinue, eSelect, aNum, aSelect, net, p):
        # eSelect指专家在类别发现之后的选择策略, 1表示有筛选随机, 2表示自定义策略
        # aNum指业余者在每个batch中标注数目, 1表示batchSize*accuracy_last_time, 2表示固定大小
        # aSelect指业余者数据选择策略, 1表示各类别选择一张, 2表示第一个类别选择aNum张
        # p表示业余者正确率
        if isContinue:
            self.continue_from_middle()
        if k==1: 
            self.firstBatch(self.dataset+net,p)
        else:
            self.restBatch(eSelect, aNum, aSelect,p)
        if self.left<=0:
            return False
        if len(self.reallabel)==self.n: # 若成本还没有用完就已经标完全部数据集
            return False
        self.saveMiddle(k)
        return True
    
    def firstBatch(self, path, p):
        first, second = self.myCluster.firstCluster(path)
        self.expertAnnotate(first)
        correct=0
        for clas,idx in enumerate(second):
            # clas为聚类标号，idx为应标注的label
            if idx in first:
                continue
            self.a_num += 1
            self.left -= self.a
            # 加噪后判断业余者是否返回正确答案
            reallabel_flag=False
            noise_answer=random.random()
            if self.label[idx]==self.label[first[clas]]:
                if noise_answer<=p:
                    reallabel_flag=True
                    self.a_correct += 1
                    self.a_tp+=1
                else:
                    self.a_fn+=1
            else:
                if noise_answer>p:
                    reallbel_flag=True
                    self.a_fp+=1
                else:
                    self.a_tn+=1
            if reallabel_flag:
                correct+=1
                self.reallabelDic[self.annotation[first[clas]]].append(idx)
                self.reallabel.append(idx)
                self.annotation[idx] = self.annotation[first[clas]]
            else:
                self.halflabel[idx]=[self.annotation[first[clas]]]
            #print(idx)
            self.unlabel.remove(idx)
        # 设置上一轮已发现类别个数prev
        # 设置持续未发现新类别的轮数no_progress
        # 设置长尾现象标志为False
        self.prev=len(self.discoveredClass)
        self.no_progress=0
        self.tail_flag=False
        # 统计上一轮业余者正确率
        self.lastAccuracy=correct/(len(second))

    def restBatch(self, eSelect, aNum, aSelect, p):
        self.updateModel()
        a_annotate=self.genAmateurAnnotate(aNum, aSelect)
        if p<1.0:
            self.amateurNoiseAnnotate(a_annotate,p)
        else:
            self.amateurAnnotate(a_annotate)
        e_annotate=self.genExpertAnnotate(a_annotate, eSelect) # 专家在本batch中不重复标amateur标过的图片
        self.expertAnnotate(e_annotate)
        # if p<1.0:
        #     self.amateurNoiseAnnotate(a_annotate,p)
        # else:
        #     self.amateurAnnotate(a_annotate)
        # 判断类别发现是否开始有长尾现象
        if len(self.discoveredClass)<self.l:
            self.isTail()
            print('no progress:'+str(self.no_progress)+'; isTail:'+str(self.tail_flag))

    def genAmateurAnnotate(self, aNum, aSelect):
        # 找到每个已知类别应该标注的样本，顺便计算门槛值，返回amateur标注的id列表
        a_idlist=[]
        # num=int(np.ceil(self.batch_size*self.lastAccuracy)) if int(np.ceil(self.batch_size*self.lastAccuracy))!=0 else (10 if self.n-len(self.reallabel)>10 else 0)
        num=min((self.batch_size-int(np.ceil(self.batch_size*self.lastAccuracy)))*self.ae_ratio,len(self.unlabel)+len(self.halflabel))
        #num=min(self.batch_size,len(self.unlabel)+len(self.halflabel))
        if num==0:
            return a_idlist
        #num=min(num,len(self.myCluster.amateur_rank))
        count=0
        rank_margin = num % len(self.myCluster.amateur_rank)
        if aSelect==1:
            # 每个类别抽一张
            # for label in self.myCluster.amateur_rank:
            #     # 或许存在有的类别没有样本在其上置信度最大的情况，所以要判断classification[label]是否为空
            #     if len(self.myCluster.classification[label]):
            #         self.myCluster.classification[label] = sorted(self.myCluster.classification[label], key=lambda x:abs(x[0]-0.5))
            #         a_idlist.append(self.myCluster.classification[label][0][1])
            #         count+=1
            #         if count==num:
            #             break
            # 每个类别抽多张
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
        # 剔除门槛值大于相应threshold的点，在剩余点中随机选择expert应该标的点，返回expert应该标注的id列表
        legal=[]
        print("lastAccuracy:"+str(self.lastAccuracy))
        print("batchsize:"+str(self.batch_size))
        num=self.batch_size-int(np.ceil(self.batch_size*self.lastAccuracy))
        print("expert_num:"+str(num))
        # num=int(self.batch_size-len(a_annotate))
        # 若是纯随机情况，或者类别未发现完全且为长尾现象
        if eSelect==6 or (self.tail_flag and len(self.discoveredClass)<self.l):
            print("6 or class discover random select")
            # legal=list(set(list(self.halflabel.keys())+self.unlabel)-set(a_annotate))
            legal=list(self.halflabel.keys())+self.unlabel
            return random.sample(legal, num) if num<len(legal) else legal
        # 若类别已发现完全
        if len(self.discoveredClass)==self.l: # 若类别发现结束
            if eSelect==2: # 方差大的类别依次选择
                count=0
                for label in self.myCluster.expert_rank:
                    threshold=self.myCluster.classification[label][0].confidence if len(self.myCluster.classification[label]) else 0.5
                    for item in self.myCluster.classification[label][::-1]:
                        if item[0]<=threshold and item[1] not in a_annotate:
                            legal.append(item[0])
                            count+=1
                            if count==num:
                                break
                    if count==num:
                        break
                return legal
            if eSelect==3: # 从样本最少的类别标起
                count=0
                for label in self.myCluster.num_rank:
                    if len(self.myCluster.classification[label]):
                        threshold=self.myCluster.classification[label][0][0]
                        for item in self.myCluster.classification[label]:
                            if item[0]<=threshold and item[1] not in a_annotate:
                                legal.append(item[1])
                                count+=1
                                break
                    if count==num:
                        break
                return legal
            if eSelect==4: # 方差大的类别一个选一张
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
            if eSelect==5: # 剩下所有的样本熵降序排列
                count=0
                entrophy=sorted(self.myCluster.entrophy, key=lambda x:-x[0])
                for item in entrophy:
                    if item[1] not in a_annotate:
                        legal.append(item[1])
                        count+=1
                        if count==num:
                            break
                return legal
        # 若类别未发现完全但不为长尾现象
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
        self.myCluster.restCluster(self.reallabelDic, self.halflabel, self.reallabel, self.unlabel)
        self.myCluster.rankClass(self.reallabelDic)
    
    def amateurNoiseAnnotate(self, idlist, p):
        correct=0
        for idx in idlist:
            self.left -= self.a
            self.a_num += 1
            i = np.argmax(self.myCluster.confidence[idx])
            noise_answer=random.random()
            reallabel_flag=False
            if self.label[idx]==self.discoveredClass[i]:
                if noise_answer<=p:
                    reallabel_flag=True
                    self.a_correct += 1
                    self.a_tp+=1
                else:
                    self.a_fn+=1
            else:
                if noise_answer>p:
                    reallabel_flag=True
                    self.a_fp+=1
                else:
                    self.a_tn+=1
            if reallabel_flag:
                correct += 1
                self.reallabel.append(idx)
                self.reallabelDic[i].append(idx)
                self.annotation[idx]=i
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
            self.ae_ratio=max(15,self.ae_ratio+1)
        else:
            self.ae_ratio=min(10,self.ae_ratio-1)
        self.lastAccuracy= now_Accuracy

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
                self.reallabelDic[i].append(idx)
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
            self.ae_ratio=max(15,self.ae_ratio+1)
        else:
            self.ae_ratio=min(10,self.ae_ratio-1)
        self.lastAccuracy= now_Accuracy
        res1=np.argsort(res)

    def expertAnnotate(self, idlist):
        for idx in idlist:
            self.left -= self.e
            self.e_num += 1
            if len(self.discoveredClass)<self.l and self.label[idx] not in self.discoveredClass:
                self.discoveredClass.append(self.label[idx])
                self.reallabelDic[len(self.discoveredClass)-1]=[idx]
                self.reallabel.append(idx)
            else:
                i = self.discoveredClass.index(self.label[idx])
                self.reallabelDic[i].append(idx)
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

if __name__=='__main__':
    print("i am new")