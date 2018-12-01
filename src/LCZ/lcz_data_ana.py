# -*- coding: utf-8 -*-
'''
Created on 2018年11月19日

@author: zwp12
'''


'''
LCZ 数据分析
1.sen1的最大，最小属性值
2.sen2的最大，最小属性值
3.训练集中各个类型数量
'''

import numpy as np;
import time;
import h5py;
from tools import SysCheck


base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'


train_path = base_path+'/Dataset/tianci/LCZ/splited/training5.0.h5'
validation_path =  base_path+'/Dataset/tianci/LCZ/validation.h5'


def run():
    
    h5f = h5py.File(validation_path);
    
    sen1 = np.array(h5f['sen1']);
    sen2 = np.array(h5f['sen2']);
    label= np.array(h5f['label']);
    
    datasize = sen1.shape[0];
    
    class_cot=np.zeros((17,));
    
    
    sen1maxmin=[-100,100];
    sen2maxmin=[-100,100];
    for i in range(datasize):
        d1 = sen1[i];
        d2 = sen2[i];
        l = label[i];
        
        sen1maxmin[0]=max(sen1maxmin[0],np.max(d1));
        sen1maxmin[1]=min(sen1maxmin[1],np.min(d1));
        sen2maxmin[0]=max(sen2maxmin[0],np.max(d2));
        sen2maxmin[1]=min(sen2maxmin[1],np.min(d2));    
        class_cot+=l;
        if i%1000==0:
            print('step%d'%i);
            print(class_cot);
            
    pass;

    print(sen1maxmin);
    print(sen2maxmin)
    print(class_cot);


def run2():
    a = [ 5068,24431,31693, 8651, 16493, 35290,  3269, 39326, 13584, 11954,
     42902,  9514,  9165, 41377,  2392,  7898, 49359];
     
    a = np.array(a)/352366;
    
    print(a/np.sum(a));
    print(np.sum(a),np.sum(a[10:])) 


if __name__ == '__main__':
    run2();
    pass

