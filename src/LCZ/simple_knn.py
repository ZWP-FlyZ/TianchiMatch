# -*- coding: utf-8 -*-
'''
Created on 2018年11月20日

@author: zwp12
'''

import numpy as np;
import time;
import h5py;
import tensorflow as tf;
from tools import SysCheck


'''

近邻分类器

'''

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
train_path = base_path+'/Dataset/tianci/LCZ/training.h5'
validation_path = base_path+'/Dataset/tianci/LCZ/validation.h5'
result_path = base_path+'/Dataset/tianci/LCZ/result.csv'
train_spa=2;
test_spa=0.1;
k=30;
# 50 - 0.31
# 30 - 0.32
# 20 - 0.31
# 10 -0.309
#  5 - 0.261

class_rat = np.array([0.01438277,0.06933416,0.08994341,0.02455118,0.04680645, 
                    0.10015155,0.00927729,0.11160555,0.03855082,0.03392495,
                    0.12175409,0.02700033,0.02600989,0.1174262 ,0.0067884 ,0.02241419,
                    0.14007878])


def knn(k,train,label,test):
    test_size = len(test);
    predict_res = [];
    for i in range(test_size):
        delta = np.sum((train-test[i])**2,axis=(1,2,3));
        idx = np.argsort(delta)[:k];
        predict_res.append(np.sum(label[idx],axis=0));
        if i%1==0:
            print('knn step%d'%i)
    return np.array(predict_res);

def tensor_sub(X,x):
    return tf.reduce_sum(tf.square(X-x),axis=(1,2,3));

def knn_tf(k,train,label,test):
#     X = tf.placeholder(tf.float32, [None,32,32,train.shape[3]]);
    X = tf.constant(train, dtype=tf.float32);
    x = tf.placeholder(tf.float32, [32,32,train.shape[3]]);
    test_size = len(test);
    print('test size%d'%test_size)
    predict_res = [];
    delta = tensor_sub(X,x);
    with tf.Session() as sess:
        for i in range(test_size):
            vdelta = sess.run(delta,{x:test[i]});
            idx = np.argsort(vdelta)[:k];
            predict_res.append(np.sum(label[idx],axis=0));
            if i%1==0:
                print('knn step%d'%i)    
    return np.array(predict_res);
    

def reg(x):
    minarr = np.min(x,axis=(0,1));
    maxarr = np.max(x,axis=(0,1));
    a = (x-minarr);
    b = (maxarr-minarr);
    return np.divide(a,b,out=np.zeros_like(a),where=b!=0);


def regtoOne(X):
    return np.array(list(map(reg,X)));


def random_idx(datasize,train_spa,test_spa):
    train_size = int(train_spa/100.0*datasize);
    test_size = int(test_spa/100.0*datasize);
    
    idx=np.arange(datasize,dtype=np.int);
    np.random.shuffle(idx);
    train_idx=idx[:train_size];
    test_idx = idx[datasize-test_size:];
    
    train_idx=np.sort(train_idx);
    test_idx=np.sort(test_idx);    
    
    return train_idx,test_idx;
    


def run(train_spa,test_spa):

    
    
    obj = h5py.File(train_path);
    train_sen1 = obj['sen1']
#     train_sen1 = np.array(obj['sen1']);
    datasize = train_sen1.shape[0];
    train_idx,test_idx = random_idx(datasize,train_spa,test_spa);
    train_sen1 = np.array(train_sen1[train_idx.tolist()]);
    train_sen2 = np.array(obj['sen2'][train_idx.tolist()]);
    train_label = np.array(obj['label'][train_idx.tolist()]);
    
    test_sen1 = np.array(obj['sen1'][test_idx.tolist()]);
    test_sen2 = np.array(obj['sen2'][test_idx.tolist()]);
    test_label = np.array(obj['label'][test_idx.tolist()]);    
    
    #  归一化
    # 一张图片内各个特征进行归一化
    train_sen1 = regtoOne(train_sen1);
    train_sen2 = regtoOne(train_sen2);
    test_sen1 = regtoOne(test_sen1);
    test_sen2 = regtoOne(test_sen2);
    
    res_sen1 = knn_tf(k,train_sen1,train_label,test_sen1);
    res_sen2 = knn_tf(k,train_sen2,train_label,test_sen2);
    
    res = (res_sen1+res_sen2)
    res = res/np.sum(res,axis=1,keepdims=True);
    print(res);
    res = res*class_rat
    print(res);
    py=np.argmax(res,axis=1);
    y = np.argmax(test_label,axis=1);
    print(py);
    print(y);
    ss = len(py);
    es = np.count_nonzero(py-y);
    print(ss,es);
    print((ss-es)*1.0/ss);
    
    res_out = np.zeros((len(py),17),np.int);
    res_out[np.arange(len(py)),py]=1;
    print(res_out);
    np.savetxt(result_path,res_out,'%d', delimiter=',');
    
    pass;

def run_validation(train_spa,test_spa):

    
    
    obj = h5py.File(train_path);
    objva = h5py.File(validation_path);
    train_sen1 = obj['sen1']
#     train_sen1 = np.array(obj['sen1']);
    datasize = train_sen1.shape[0];
    train_idx,_ = random_idx(datasize,train_spa,test_spa);
    train_sen1 = np.array(train_sen1[train_idx.tolist()]);
    train_sen2 = np.array(obj['sen2'][train_idx.tolist()]);
    train_label = np.array(obj['label'][train_idx.tolist()]);
    
    test_sen1 = np.array(objva['sen1']);
    test_sen2 = np.array(objva['sen2']);

    
    #  归一化
    # 一张图片内各个特征进行归一化
    train_sen1 = regtoOne(train_sen1);
    train_sen2 = regtoOne(train_sen2);
    test_sen1 = regtoOne(test_sen1);
    test_sen2 = regtoOne(test_sen2);
    
    res_sen1 = knn_tf(k,train_sen1,train_label,test_sen1);
    res_sen2 = knn_tf(k,train_sen2,train_label,test_sen2);
    
    res = (res_sen1+res_sen2)
    res = res/np.sum(res,axis=1,keepdims=True);
    print(res);
    res = res*class_rat
    print(res);
    py=np.argmax(res,axis=1);
    
    
    res_out = np.zeros((len(py),17),np.int);
    res_out[np.arange(len(py)),py]=1;
    print(res_out);
    np.savetxt(result_path,res_out,'%d', delimiter=',');
    

    print(py);


    
    #### 存储 
    
    res_out = np.zeros((len(py),17),np.int);
    res_out[np.arange(len(py)),py]=1;
    print(res_out);
    np.savetxt(result_path,res_out,'%d', delimiter=',');
    
    
    pass;    

def run_(train_spa,test_spa):
    train_path = base_path+'/Dataset/tianci/LCZ/splited/training%.1f.h5'%train_spa
    test_path = base_path+'/Dataset/tianci/LCZ/splited/test%.1f.h5'%test_spa
    
    
    obj = h5py.File(train_path);
    train_sen1 = np.array(obj['sen1']);
    train_sen2 = np.array(obj['sen2']);
    train_label = np.array(obj['label']);
    
    obj = h5py.File(test_path);
    test_sen1 = np.array(obj['sen1']);
    test_sen2 = np.array(obj['sen2']);
    test_label = np.array(obj['label']);    
    
    #  归一化
    # 一张图片内各个特征进行归一化
    train_sen1 = regtoOne(train_sen1);
    train_sen2 = regtoOne(train_sen2);
    test_sen1 = regtoOne(test_sen1);
    test_sen2 = regtoOne(test_sen2);
    
    res_sen1 = knn_tf(k,train_sen1,train_label,test_sen1);
    res_sen2 = knn_tf(k,train_sen2,train_label,test_sen2);
    
    res = (res_sen1+res_sen2)
    res = res/np.sum(res,axis=1,keepdims=True);
    print(res);
    res = res*class_rat
    print(res);
    py=np.argmax(res,axis=1);
    y = np.argmax(test_label,axis=1);
    print(py);
    print(y);
    ss = len(py);
    es = np.count_nonzero(py-y);
    print(ss,es);
    print((ss-es)*1.0/ss);
    pass;



if __name__ == '__main__':
#     run(train_spa,test_spa);
    run_validation(train_spa,test_spa);
#     a = np.random.normal(size=(1,3,3,2));
#     print(a);
#     print(np.array(list(map(reg,a))));
    
    
    pass