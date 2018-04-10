# -*- coding: utf-8 -*-
'''
Created on 2018年4月9日

@author: zwp
'''

import numpy as np;

def dist2(X,tX):
    tX  = np.reshape(tX,[tX.shape[0],1,tX.shape[1]]);
    result = np.sqrt(np.sum((X-tX)**2,axis=2));
    return result;
    pass;

a = np.arange(12).reshape([4,3]);
b = np.arange(6).reshape([2,3]);
c = np.arange(10).reshape([-1,1]);
print(a,'\n',b);
# b = b.reshape([2,1,3])

dis = dist2(a,b);
print(dis);


indx = np.argsort(dis, axis=1);

print(np.max(a,axis=1));



if __name__ == '__main__':
    pass