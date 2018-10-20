# -*- coding: utf-8 -*-
'''
Created on 2018年10月20日

@author: zwp12
'''
from tools.Normal import tranf2Guass

'''
k-nn算法
'''
import numpy as np;
from tools.Normal import *;
from tools.DimReduce import pca;
from tools import SysCheck

def knn(trian,test,k):
    train_x,train_y = trian;
    test_x,_ = test;
    
    
    py = [];
    for i in range(len(test_x)):
        tx = test_x[i];
        delta = np.sum((train_x - tx)**2,axis=1);
        topkidx =np.argsort(delta)[:k];
        v = np.zeros([4],np.int);
        for idx in topkidx:
            tmp = train_y[idx];
            v[tmp] = v[tmp]+1;
        ridx = np.argmax(v);
        py.append(ridx);
        if i%20==0:
            print('step=%d finished!'%i);
    return np.array(py);
    

def evel(y,py):
    datasize = len(y);
    YT=np.zeros([4])
    TP=np.zeros([4])
    PYT=np.zeros([4])
    for i in range(datasize):
        yv = y[i];
        pyv =py[i];
        YT[yv]+=1;
        PYT[pyv]+=1;
        if yv == pyv:
            TP[yv]+=1;
    print('YT=\t',YT);    
    print('TP=\t',TP);
    print('PYT=\t',PYT);
    pre = TP/PYT;
    recall = TP/YT;
    f1 = 2*pre*recall / (pre+recall)
    print('pre=\t',TP/PYT);
    print('recall=\t',TP/YT);
    print('f1=\t',f1);
    print('F1=\t',np.mean(f1));

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work';
base_path=base_path+r'/Dataset/tianci/LAMOST2';
train_path = base_path+'/data_2600/train'
test_path = base_path+'/data_2600/test'


def run():
    train_x = np.loadtxt(train_path+'/x.txt',np.float);
    train_y = np.loadtxt(train_path+'/y.txt',np.int);
    
    test_x = np.loadtxt(test_path+'/x.txt',np.float);
    test_y = np.loadtxt(test_path+'/y.txt',np.int);
    print('load data finished!');
    
    train_x = tranf2One2(train_x,1);
    test_x = tranf2One2(test_x,1);
     
#     train_x = tranf2Center(train_x,0);
#     test_x = tranf2Center(test_x,0);
 
     
#     train_x = tranf2Guass(train_x,0);
#     test_x = tranf2Guass(test_x,0);
     
    train_x,test_x = pca(train_x,test_x,1000);     
    
    
    
    k=10
    py = knn((train_x,train_y),(test_x,test_y),k);
    print(py);
    evel(test_y,py);
    
    
    pass;

if __name__ == '__main__':
    run();
    pass