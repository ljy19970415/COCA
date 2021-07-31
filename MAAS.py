# -*- coding: UTF-8 -*-
from annotate import Annotate
from cluster import Cluster
import numpy as np
import math
import time
import csv
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def append_csv(path,datas):
    with open(path, "a", newline='') as file: # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        csv_file.writerows(datas)

def append_txt(path,datas):
    f = open(path, mode='a',encoding='utf-8')
    print(datas, file = f)
    f.close()

def generateTrain(annotate, subtitle, rootpath):
    if not os.path.exists(rootpath + subtitle):
        os.mkdir(rootpath + subtitle)
    np.save(rootpath+subtitle+'/reallabel.npy', annotate.reallabel)
    np.save(rootpath+subtitle+'/halflabel.npy', annotate.halflabel)
    np.save(rootpath+subtitle+'/unlabel.npy', annotate.unlabel)
    np.save(rootpath+subtitle+'/info.npy', [annotate.B-annotate.left, annotate.e_num, annotate.a_num, annotate.a_correct])
    np.save(rootpath+subtitle+'/discoveredClass.npy',annotate.discoveredClass)

def MAAS(dataset, e, a, isEqual, isContinue, eSelect, aNum, aSelect, net, p):
    # mode 为使用的utility计算方法
    if p<1:
        rootpath=dataset[:-1]+'_records_noise/'+net+'_'+str(eSelect)+'/'
    else:
        rootpath=dataset[:-1]+'_records/'+net+'_'+str(eSelect)+'/'
    print(rootpath)
    # 相同成本比精度, cost_a_b, k=a, cost=b% expert annotation
    cost_5_1,cost_10_1,cost_20_1,cost_30_1=True,True,True,True
    cost_5_2,cost_10_2,cost_20_2,cost_30_2=True,True,True,True
    cost_5_3,cost_10_3,cost_20_3,cost_30_3=True,True,True,True
    cost_5_4,cost_10_4,cost_20_4,cost_30_4=True,True,True,True
    cost_5_5,cost_10_5,cost_20_5,cost_30_5=True,True,True,True
     
    annotate = Annotate(dataset, e, a, rootpath, net) 
    k=1
    flag=True
    time_start = time.time()
    if isContinue:
        k=np.load(rootpath+'middle_point/k.npy')[0]+1
        flag=annotate.annotate(k, True, eSelect, aNum, aSelect, net, p)
        precision, cost, class_num, e_num, a_num, a_correct, min_num, max_num, std=annotate.calBatchPrecision()
        print("batch: "+str(k)+"; precision: "+str(precision)+"; cost: "+str(cost)+"; class_num: "+str(class_num)+"; min: "+str(min_num)+"; max: "+str(max_num)+"; std: "+str(std))
        print("e_num: "+str(e_num)+"; a_num: "+str(a_num)+"; a_correct: "+str(a_correct)+"; total_correct: "+str(a_correct+e_num))
        append_csv(rootpath+'result.csv',[[k, precision, cost, e_num, a_num, a_correct, a_correct+e_num, std, class_num]])
        append_txt(rootpath+'result.txt',",".join([str(k), str(precision), str(cost), str(e_num), str(a_num), str(a_correct), str(a_correct+e_num), str(std), str(class_num)]))
        k+=1
    while flag:
        flag=annotate.annotate(k, False, eSelect, aNum, aSelect, net, p)
        precision, cost, class_num, e_num, a_num, a_correct, min_num, max_num, std=annotate.calBatchPrecision()
        print("batch: "+str(k)+"; precision: "+str(precision)+"; cost: "+str(cost)+"; class_num: "+str(class_num)+"; min: "+str(min_num)+"; max: "+str(max_num)+"; std: "+str(std))
        print("e_num: "+str(e_num)+"; a_num: "+str(a_num)+"; a_correct: "+str(a_correct)+"; total_correct: "+str(a_correct+e_num))
        #append_csv(rootpath+'result.csv',[[k, precision, cost, e_num, a_num, annotate.a_tp,annotate.a_fn,annotate.a_tn,annotate.a_fp,annotate.a_tp+e_num, len(annotate.reallabel),std, class_num]])
        append_csv(rootpath+'result.csv',[[k, precision, cost, e_num, a_num, a_correct, a_correct+e_num, std, class_num]])
        append_txt(rootpath+'result.txt',",".join([str(k), str(precision), str(cost), str(e_num), str(a_num), str(a_correct), str(a_correct+e_num), str(std), str(class_num)]))
        k+=1
        if cost_5_5 and (e_num+a_num//5)>=math.ceil(annotate.n*0.5):
            cost_5_5 = False
            generateTrain(annotate, '5_5', rootpath)
        elif cost_5_4 and (e_num+a_num//5)>=math.ceil(annotate.n*0.4):
            cost_5_4 = False
            generateTrain(annotate, '5_4', rootpath)
        elif cost_5_3 and (e_num+a_num//5)>=math.ceil(annotate.n*0.3):
            cost_5_3 = False
            generateTrain(annotate, '5_3', rootpath)
        elif cost_5_2 and (e_num+a_num//5)>=math.ceil(annotate.n*0.2):
            cost_5_2 = False
            generateTrain(annotate, '5_2', rootpath)
        elif cost_5_1 and (e_num+a_num//5)>=math.ceil(annotate.n*0.1):
            cost_5_1 = False
            generateTrain(annotate, '5_1', rootpath)
        
        if cost_10_5 and (e_num+a_num//10)>=math.ceil(annotate.n*0.5):
            cost_10_5 = False
            generateTrain(annotate, '10_5', rootpath)
        elif cost_10_4 and (e_num+a_num//10)>=math.ceil(annotate.n*0.4):
            cost_10_4 = False
            generateTrain(annotate, '10_4', rootpath)
        elif cost_10_3 and (e_num+a_num//10)>=math.ceil(annotate.n*0.3):
            cost_10_3 = False
            generateTrain(annotate, '10_3', rootpath)
        elif cost_10_2 and (e_num+a_num//10)>=math.ceil(annotate.n*0.2):
            cost_10_2 = False
            generateTrain(annotate, '10_2', rootpath)
        elif cost_10_1 and (e_num+a_num//10)>=math.ceil(annotate.n*0.1):
            cost_10_1 = False
            generateTrain(annotate, '10_1', rootpath)
        
        if cost_20_5 and (e_num+a_num//20)>=math.ceil(annotate.n*0.5):
            cost_20_5 = False
            generateTrain(annotate, '20_5', rootpath)
        elif cost_20_4 and (e_num+a_num//20)>=math.ceil(annotate.n*0.4):
            cost_20_4 = False
            generateTrain(annotate, '20_4', rootpath)
        elif cost_20_3 and (e_num+a_num//20)>=math.ceil(annotate.n*0.3):
            cost_20_3 = False
            generateTrain(annotate, '20_3', rootpath)
        elif cost_20_2 and (e_num+a_num//20)>=math.ceil(annotate.n*0.2):
            cost_20_2 = False
            generateTrain(annotate, '20_2', rootpath)
        elif cost_20_1 and (e_num+a_num//20)>=math.ceil(annotate.n*0.1):
            cost_20_1 = False
            generateTrain(annotate, '20_1', rootpath)
        
        if cost_30_5 and (e_num+a_num//30)>=math.ceil(annotate.n*0.5):
            cost_30_5 = False
            generateTrain(annotate, '30_5', rootpath)
        elif cost_30_4 and (e_num+a_num//30)>=math.ceil(annotate.n*0.4):
            cost_30_4 = False
            generateTrain(annotate, '30_4', rootpath)
        elif cost_30_3 and (e_num+a_num//30)>=math.ceil(annotate.n*0.3):
            cost_30_3 = False
            generateTrain(annotate, '30_3', rootpath)
        elif cost_30_2 and (e_num+a_num//30)>=math.ceil(annotate.n*0.2):
            cost_30_2 = False
            generateTrain(annotate, '30_2', rootpath)
        elif cost_30_1 and (e_num+a_num//30)>=math.ceil(annotate.n*0.1):
            cost_30_1 = False
            generateTrain(annotate, '30_1', rootpath)
    time_end = time.time()
    print("annotation cost "+str(time_end-time_start//60)+" min")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l',type=int, default=50)
    parser.add_argument('--e', type=int, default=10)
    parser.add_argument('--a', type=int, default=1)
    parser.add_argument('--isEqual', type=int, default=0)
    parser.add_argument('--isContinue', type=int, default=0)
    parser.add_argument('--eSelect', type=int, default=4)
    parser.add_argument('--aNum', type=int, default=1)
    parser.add_argument('--aSelect', type=int, default=1)
    parser.add_argument('--net', type=str, default='vgg16')
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default="Stanford Dogs")

    args = parser.parse_args()

    #dataset_path='datasets/'+str(args.isEqual)+'_'+str(args.l)+'/'
    #MAAS(dataset_path, args.l, args.e, args.a, args.isEqual, args.isContinue, args.eSelect, args.aNum, args.aSelect, args.net)
    #MAAS('datasets/1_200/', 200, args.e, args.a, 1, args.isContinue, 4, args.aNum, args.aSelect, 'resnet50', args.p)
    MAAS(args.dataset+'/', args.e, args.a, 1, args.isContinue, 4, args.aNum, args.aSelect, 'resnet50', args.p)
