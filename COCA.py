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

def COCA(dataset, e, a, isEqual, isContinue, eSelect, aNum, aSelect, net):
    # mode 为使用的utility计算方法
    if p<1:
        rootpath=dataset[:-1]+'_records_noise/'+net+'_'+str(eSelect)+'/'
    else:
        rootpath=dataset[:-1]+'_records/'+net+'_'+str(eSelect)+'/'
    print(rootpath)
     
    annotate = Annotate(dataset, e, a, rootpath, net) 
    k=1
    flag=True
    time_start = time.time()
    if isContinue:
        k=np.load(rootpath+'middle_point/k.npy')[0]+1
        flag=annotate.annotate(k, True, eSelect, aNum, aSelect, net)
        precision, cost, class_num, e_num, a_num, a_correct, min_num, max_num, std=annotate.calBatchPrecision()
        print("batch: "+str(k)+"; precision: "+str(precision)+"; cost: "+str(cost)+"; class_num: "+str(class_num)+"; min: "+str(min_num)+"; max: "+str(max_num)+"; std: "+str(std))
        print("e_num: "+str(e_num)+"; a_num: "+str(a_num)+"; a_correct: "+str(a_correct)+"; total_correct: "+str(a_correct+e_num))
        append_csv(rootpath+'result.csv',[[k, precision, cost, e_num, a_num, a_correct, a_correct+e_num, std, class_num]])
        append_txt(rootpath+'result.txt',",".join([str(k), str(precision), str(cost), str(e_num), str(a_num), str(a_correct), str(a_correct+e_num), str(std), str(class_num)]))
        k+=1
    while flag:
        flag=annotate.annotate(k, False, eSelect, aNum, aSelect, net)
        precision, cost, class_num, e_num, a_num, a_correct, min_num, max_num, std=annotate.calBatchPrecision()
        print("batch: "+str(k)+"; precision: "+str(precision)+"; cost: "+str(cost)+"; class_num: "+str(class_num)+"; min: "+str(min_num)+"; max: "+str(max_num)+"; std: "+str(std))
        print("e_num: "+str(e_num)+"; a_num: "+str(a_num)+"; a_correct: "+str(a_correct)+"; total_correct: "+str(a_correct+e_num))
        #append_csv(rootpath+'result.csv',[[k, precision, cost, e_num, a_num, annotate.a_tp,annotate.a_fn,annotate.a_tn,annotate.a_fp,annotate.a_tp+e_num, len(annotate.reallabel),std, class_num]])
        append_csv(rootpath+'result.csv',[[k, precision, cost, e_num, a_num, a_correct, a_correct+e_num, std, class_num]])
        append_txt(rootpath+'result.txt',",".join([str(k), str(precision), str(cost), str(e_num), str(a_num), str(a_correct), str(a_correct+e_num), str(std), str(class_num)]))
        k+=1
    time_end = time.time()
    print("annotation cost "+str(time_end-time_start//60)+" min")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l',type=int, default=50)
    parser.add_argument('--e', type=int, default=10)
    parser.add_argument('--a', type=int, default=1)
    parser.add_argument('--isEqual', type=int, default=1)
    parser.add_argument('--isContinue', type=int, default=0)
    parser.add_argument('--eSelect', type=int, default=4)
    parser.add_argument('--aNum', type=int, default=1)
    parser.add_argument('--aSelect', type=int, default=1)
    parser.add_argument('--net', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default="Stanford Dogs")

    args = parser.parse_args()

    # you only have to adjust parameter: dataset, net, isContinue
    # dataset can be either 'CUB' or 'Stanford Dogs'
    # net can be 'resnet50', 'vgg16' or 'mobilenet'
    # isContinue means start from the middle point
    COCA(args.dataset+'/', args.e, args.a, args.isEqual, args.isContinue, args.eSelect, args.aNum, args.aSelect, args.net)
