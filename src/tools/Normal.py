# -*- coding: utf-8 -*-
'''
Created on 2018年10月20日

@author: zwp12
'''

import numpy as np;

def tranf2One(x,axis=0):
    '''
    归一化
    '''

    mi = np.min(x,axis=axis);
    ma = np.max(x,axis=axis);
    if axis==1:
        mi =mi.reshape([-1,1]);
        ma = ma.reshape([-1,1]);
    return (x-mi)/(ma-mi);


def tranf2One2(x,axis=0):
    '''
    归一化
    '''

    mi = np.min(x,axis=axis);
    ma = np.max(x,axis=axis);
    mean  = np.mean(x,axis=axis);
    if axis==1:
        mi =mi.reshape([-1,1]);
        ma = ma.reshape([-1,1]);
        mean = mean.reshape([-1,1]);
    return (x-mean)/(ma-mi);


def tranf2Guass(x,axis=0):
    '''
    标准正态化
    '''
    mean = np.mean(x,axis=axis);
    std = np.std(x,axis=axis);
    if axis==1:
        mean = mean.reshape([-1,1])
        std = std.reshape([-1,1])
    return (x-mean)/std;

def tranf2Center(x,axis=0):
    '''
    特征中心化
    '''
    mean = np.mean(x,axis=axis);
    if axis==1:
        mean = mean.reshape([-1,1])
    return x-mean;
    
    

if __name__ == '__main__':
    
    
    a = np.arange(16).reshape(4,4)
    print(a);
    print(tranf2One(a,0))
    print(tranf2Guass(a,0))
    print(tranf2Center(a))
    
    pass