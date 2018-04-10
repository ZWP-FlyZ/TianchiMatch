# -*- coding: utf-8 -*-
'''
Created on 2018年4月8日

@author: zwp
'''

import numpy as np;


def normal(X):
    rk = np.ndim(X);
    if rk == 1:
        X = np.reshape(X,[-1,X.shape[0]]);
    max_xs = np.reshape(np.max(X,axis=1),[-1,1]);
    min_xs = np.reshape(np.min(X,axis=1),[-1,1]);
    X = (X - min_xs) / (max_xs-min_xs);
    if rk == 1:
        X = X.flatten();
    return X;

def dist(X,x):
    '''
    X:[batch,feature_x]的训练集
    x:[feature_x]测试数据项
    return [batch] 距离值
    '''

    # 欧式距离法
    result = np.sqrt(np.sum((X-x)**2,axis=1));

    return result;

a = np.array([1,2,3]);
b = np.reshape(np.random.normal(size=9),[3,3]);
b = np.reshape(np.arange(9),[3,3]);

print(np.max(b,axis=1));

print(np.ndim(a),np.ndim(b))


print(b);
print(normal(a));

d = dist(b,a);
print(d);
dso = np.argsort(d);
print(np.where(a==1));






if __name__ == '__main__':
    pass