# -*- coding: utf-8 -*-
'''
Created on 2018年10月20日

@author: zwp12
'''

'''

利用PCA和T-SNE进行降维


'''

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np;


def pca(train_x,test_x=None,dim=2):
    '''
    PCA 线性降维
    p=True,进行中心化
    '''
    p = PCA(n_components=dim);
    trainx = p.fit_transform(train_x);
    testx=None;
    if not test_x is  None:
        testx = p.transform(test_x);
    return trainx,testx;

def tsne(train_x,test_x,dim=2):
    '''
    T-SNE 非线性降维
    '''
    x  = np.concatenate([train_x,test_x],axis=0);
    t = TSNE(n_components=dim);
    nx = t.fit_transform(x);
    ts = len(train_x);
    return nx[:ts],nx[ts:];


if __name__ == '__main__':
    pass