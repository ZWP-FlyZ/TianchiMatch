# -*- coding: utf-8 -*-
'''
Created on 2018年11月19日

@author: zwp12
'''


'''
数据集划分
'''


import numpy as np;
import time;
import h5py;
from tools import SysCheck


base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'


train_path = base_path+'/Dataset/tianci/LCZ/training.h5'
validation_path = base_path+'/Dataset/tianci/LCZ/validation.h5'

out_path = base_path+'/Dataset/tianci/LCZ/splited'


train_spa=3;
test_spa=0.6;


def run_spto10set(in_path,out_path,isTraining=True):
    in_ds = h5py.File(in_path,'r');
    ori_sen1 = in_ds['sen1'];
    ori_sen2 = in_ds['sen2'];
    ori_label = in_ds['label'];
    
    datasize = ori_sen1.shape[0];
    train_size = int(0.1*datasize);
        
    idx=np.arange(datasize,dtype=np.int);
    np.random.shuffle(idx);
    
    for i in range(10):

        # 去除余下物品
        start=i*train_size;
        end=min(start+train_size,datasize);
        tag_idx = np.sort(idx[start:end]).tolist();
        
        if isTraining :
            path = out_path+'/training%d.h5'%(i+1);
        else:
            path = out_path+'/test%d.h5'%(i+1);
        train_out = h5py.File(path,'w');
        train_sen1 = train_out.create_dataset('sen1', (end-start,32,32,8), 'f8');
        train_sen2 = train_out.create_dataset('sen2', (end-start,32,32,10), 'f8');
        train_label = train_out.create_dataset('label', (end-start,17), 'f8');
        

        step=100;
        e_size = len(tag_idx);
        cot = np.ceil(e_size/step);
        print(datasize,e_size,cot);
        for j in range(int(cot)):
            mic_sta = j*step;
            mic_end = min(mic_sta+step,e_size);
            step_idx = tag_idx[mic_sta:mic_end];
            train_sen1[mic_sta:mic_end,:,:,:]=ori_sen1[step_idx];
            train_sen2[mic_sta:mic_end,:,:,:]=ori_sen2[step_idx];
            train_label[mic_sta:mic_end,:]=ori_label[step_idx];
            print('dataset%d-step%d'%(i+1,j));
        
    

def run(train_spa,test_spa):
    train_out_path = base_path+'/Dataset/tianci/LCZ/splited/training%.1f.h5'%train_spa;
    test_out_path = base_path+'/Dataset/tianci/LCZ/splited/test%.1f.h5'%test_spa
    in_ds = h5py.File(train_path,'r');
    train_out = h5py.File(train_out_path,'w');
    test_out = h5py.File(test_out_path,'w');
    
    ori_sen1 = in_ds['sen1'];
    ori_sen2 = in_ds['sen2'];
    ori_label = in_ds['label'];
    datasize = ori_sen1.shape[0];
    train_size = int(train_spa/100.0*datasize);
    test_size = int(test_spa/100.0*datasize);
    
    idx=np.arange(datasize,dtype=np.int);
    np.random.shuffle(idx);
    train_idx=idx[:train_size];
    test_idx = idx[datasize-test_size:];
    
    train_idx=np.sort(train_idx);
    test_idx=np.sort(test_idx);
    print(train_idx);
    print(test_idx);
    
#     train_out.create_dataset('sen1', data=ori_sen1[train_idx.tolist()]);
# #     train_sen1 = train_out.create_dataset('sen1', (train_size,32,32,8), 'f8');
# #     train_sen1[:,:,:]=ori_sen1[train_idx.tolist()];
#     print('train sen1 finished');
#     train_out.create_dataset('sen2', data=ori_sen2[train_idx]);
#     print('train sen2 finished');
#     train_out.create_dataset('label', data=ori_label[train_idx]);
#     print('train label finished');
#     
#     
#     test_out.create_dataset('sen1', data=ori_sen1[test_idx]);
#     print('test sen1 finished');
#     test_out.create_dataset('sen2', data=ori_sen2[test_idx]);
#     print('test sen2 finished');
#     test_out.create_dataset('label', data=ori_label[test_idx]);
#     print('test label finished');      
    
    train_sen1 = train_out.create_dataset('sen1', (train_size,32,32,8), 'f8');
    train_sen2 = train_out.create_dataset('sen2', (train_size,32,32,10), 'f8');
    train_label = train_out.create_dataset('label', (train_size,17), 'f8');
     
    test_sen1 = test_out.create_dataset('sen1', (test_size,32,32,8), 'f8');
    test_sen2 = test_out.create_dataset('sen2', (test_size,32,32,10), 'f8');
    test_label = test_out.create_dataset('label', (test_size,17), 'f8');     
    
    
    for i,idx in enumerate(train_idx.tolist()):
        train_sen1[i,:,:,:]=ori_sen1[idx];
        train_sen2[i,:,:,:]=ori_sen2[idx];
        train_label[i,:]=ori_label[idx];
        if i%20==0:
            print(i);
    print('train finished')
    for i,idx in enumerate(test_idx.tolist()):
        test_sen1[i,:,:,:]=ori_sen1[idx];
        test_sen2[i,:,:,:]=ori_sen2[idx];
        test_label[i,:]=ori_label[idx];
        if i%20==0:
            print(i);
                
    
    pass;




if __name__ == '__main__':
#     run(train_spa,test_spa)
    run_spto10set(validation_path,out_path,False);
    pass