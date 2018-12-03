# -*- coding: utf-8 -*-
'''
Created on 2018年11月27日

@author: zwp12
'''

'''
利用keras 自带的app分类
'''


import numpy as np;
import time;
import h5py;
import os;
import tensorflow as tf;
from tensorflow import keras;
from tools import SysCheck
from tensorflow.keras.layers import Conv2D,AveragePooling2D,Input,Flatten,Dense,Dropout,concatenate;
from tensorflow.keras.applications import xception,vgg19,vgg19
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
result_path = base_path+'/Dataset/tianci/LCZ/result.csv'

out_path = base_path+'/Dataset/tianci/LCZ/round1_test_a_20181109.h5'

model_save_path = base_path+'/Dataset/tianci/LCZ';

case=2;
val_case=1;
learn_rate=[0.005,0.005,0.005];
batch_size=20;
epoch=[10,10,10];

input_shape=(32,32,10)
load_model=True;

def reg(x):
    minarr = np.min(x);
    maxarr = np.max(x);
    a = (x-minarr);
    b = (maxarr-minarr);
    return np.divide(a,b,out=np.zeros_like(a),where=b!=0);

    
def regtoOne(X):
    return np.array(list(map(reg,X)));

def newReg(X):
    def reg_f(x):
        return x/np.mean(x);
    return np.array(list(map(reg_f,X)));



def run():

    train_path = base_path+'/Dataset/tianci/LCZ/splited/training%d.h5'%case
    test_paht = base_path+'/Dataset/tianci/LCZ/splited/test%d.h5'%case
    
#     train_path = base_path+'/Dataset/tianci/LCZ/training.h5'
    
    train_set = h5py.File(train_path);
    test_set = h5py.File(test_paht);
    out_set = h5py.File(out_path);
#     np.random.seed(1212121);
    tag=10
    while True:
        idx = np.random.randint(0,35235);
        train_sen1 = np.array(train_set['sen2'][idx]);
        train_label = np.array(train_set['label'][idx]);
        if train_label[tag] ==1:break;
#     idx=2;

#     test_sen1 = np.array(test_set['sen2']);
#     test_label = np.array(test_set['label']);

#     train_sen1 = regtoOne(train_sen1);

    fig, axs = plt.subplots(2, 5)
    print(np.argmax(train_label));
    for row in range(2):
        for col in range(5):
            ax = axs[row, col]
            pcm = ax.pcolormesh(train_sen1[:,:,row*5+col],cmap='viridis')
    fig.colorbar(pcm, noax=axs[:, :],location='bottom')
    plt.show()
    
def run2():

    train_path = base_path+'/Dataset/tianci/LCZ/splited/training%d.h5'%case
    test_paht = base_path+'/Dataset/tianci/LCZ/splited/test%d.h5'%case
    
#     train_path = base_path+'/Dataset/tianci/LCZ/training.h5'
    
    train_set = h5py.File(train_path);
    test_set = h5py.File(test_paht);
    out_set = h5py.File(out_path);
#     np.random.seed(1212121);
    tag=6
    while True:
        idx = np.random.randint(0,35235);
        train_sen1 = np.array(train_set['sen2'][idx]);
        train_label = np.array(train_set['label'][idx]);
        if train_label[tag] ==1:break;


    train_sen1 = train_sen1.reshape((-1,10))
#     train_sen1 = train_sen1 / np.mean(train_sen1);
    np.random.shuffle(train_sen1);
    
############## 离群分析 ###################    
    
    
    v_mean = np.mean(train_sen1,axis=0,keepdims=True);
    v_std = np.std(train_sen1,axis=0,keepdims=True);
    v_rag = v_mean+2*v_std;
    v_rag = np.concatenate([v_rag,v_mean-2*v_std],axis=0);
    
    print(v_mean);
    print(v_std);
    print(v_rag);
    
    aft_train_sen = train_sen1.copy();
    for i in range(10):
        tmp = aft_train_sen[:,i];
        idxu = np.where(tmp>v_rag[0,i])
        idxd = np.where(tmp<v_rag[1,i])
        tmp[idxu]=v_mean[0,i];
        tmp[idxd]=v_mean[0,i];
        print(np.sum(tmp-aft_train_sen[:,i]));
        aft_train_sen[:,i]=tmp;
        
#     win_size=16;
#     step=8;
#     start=0;end=0;i=0;
#     res=[];
#     while end<1024:
#         start=i*step;
#         end = min(win_size+start,32*32);
#         i+=1;
#         tag = np.mean(train_sen1[start:end,:],axis=0);
#         res.append(tag);
#     aft_train_sen = np.array(res);    
        
    print(aft_train_sen);
    fig, axs = plt.subplots(2, 1)
    print(np.argmax(train_label));
    norm = mpl.colors.Normalize(vmin=np.min(train_sen1), vmax=np.max(train_sen1))
    ax = axs[0];
    pcm = ax.pcolormesh(train_sen1,cmap='viridis', norm=norm)
    fig.colorbar(pcm, ax=axs[0])
    ax = axs[1];
    pcm = ax.pcolormesh(aft_train_sen,cmap='viridis', norm=norm)    
    
    fig.colorbar(pcm, ax=axs[1])
    plt.show()    
    

if __name__ == '__main__':
    run2();
    
#     a = np.arange(27);
#     print(a);
#     b = a.reshape((1,3,3,3));
#     print(b);
#     print(b.reshape((1,9,3,1)));
    
    
    pass