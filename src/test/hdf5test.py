# -*- coding: utf-8 -*-
'''
Created on 2018年11月19日

@author: zwp12
'''

import numpy as np;
import h5py;




origin_path = 'E:/work/Dataset/tianci/LCZ'


train_path = origin_path+'/training.h5'
validation_path = origin_path+'/validation.h5'
out_path = origin_path+'/out.h5';

h5obj =  h5py.File(train_path,'r');

outh5 = h5py.File(out_path,'w');
print(h5obj);

print(list(h5obj.keys()))

sen1 = h5obj['sen1'];
sen2 = h5obj['sen2'];
label = h5obj['label'];

print(sen1);
print(sen1.shape);

print(sen2);
print(sen2.shape);

print(label);
print(label.shape);



# for idx in range(10):
#     d1 = sen1[idx];
#     d2 = sen2[idx];
#     l1 = label[idx];
#     print(np.max(d1),np.min(d1));
#     print(np.max(d2),np.min(d2));
#     print(l1);

idx = 1;
d1 = sen1[idx];
d2 = sen2[idx];
l1 = label[idx];

print(np.max(d1),np.min(d1));
print(np.max(d2),np.min(d2));
print(l1);


d1 = (d1-np.min(d1))/(np.max(d1)-np.min(d1));
d2 = (d2-np.min(d2))/(np.max(d2)-np.min(d2))

print(d1);
print(d2);


new_senw = outh5.create_dataset('sen1', shape=(10000,5), dtype='f8')
new_senw[0]=np.arange(5)
print(new_senw[[0,3,2]]);






if __name__ == '__main__':
    pass