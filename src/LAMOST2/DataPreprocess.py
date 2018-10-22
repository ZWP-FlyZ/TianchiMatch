# -*- coding: utf-8 -*-
'''
Created on 2018年10月20日

@author: zwp12
'''

'''
一些数据准备函数
'''

from LAMOST2.DataSpliter import *;
from LAMOST2.DS import  DataSource;
from tools.DimReduce import pca;
from tools.Normal import *;
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work';
base_path=base_path+r'/Dataset/tianci/LAMOST2';

zip_data_path = base_path+'/train_data.zip';

inited_ori_path = base_path+r'/train_index.txt';

oridata_path  = base_path+'/data_2600';

def oridata_spiliter():
    '''
    调用分割器，分割原始比例的数据
    '''
    '''
    star有442969（91.55%）,
    galaxy有5231（1.08%）,
    qso有1363（0.28%）,
    unknown有34288（7.08%）
    '''
    
    train_need={TYPE_STAR:4000,
                TYPE_GALAXY:2000,
                TYPE_QSO:800,
                TYPE_UNKNOWN:2000};
    
    test_need={TYPE_STAR:500,
               TYPE_GALAXY:2000,
               TYPE_QSO:500,
               TYPE_UNKNOWN:2000};
    
    random_spilter(inited_ori_path,oridata_path,
                   (train_need,test_need));
    
    pass;


def read_filedata_to_narr():
    '''
    调用数据源，将数据集提出到单文件中
    '''
    train_idx = oridata_path+'/train_data.txt';
    text_idx = oridata_path+'/test_data.txt';
    
    ds = DataSource(train_idx,zip_data_path);
    ds.saveAllToArray(oridata_path+'/train');
    ds.reloadIndex(text_idx);
    ds.saveAllToArray(oridata_path+'/test');
    
def init_origin_data():
    oridata_spiliter();
    read_filedata_to_narr();


# train2600_path= oridata_path+'/train';
# test2600_path = oridata_path+'/test';
# def TSNE_dim_reduce():
#     
#     train_x = np.loadtxt(train2600_path+'/x.txt',np.float);
#     train_y = np.loadtxt(train2600_path+'/y.txt',np.int);
#     
#     test_x = np.loadtxt(test2600_path+'/x.txt',np.float);
#     test_y = np.loadtxt(test2600_path+'/y.txt',np.int);
#     
#     dims = [3];
#     for dim in dims:
#         outpath = base_path+'/data_tsne_%d'%(dim);
# #         np.savetxt(outpath+'/train/y.txt',train_y,'%d');
# #         np.savetxt(outpath+'/test/y.txt',test_y,'%d');
#         print('save y finished')
#         trx,tex = tsne(train_x,test_x,dim);
#         print('tsne  finished');
#         np.savetxt(outpath+'/train/x.txt',trx,'%d');
#         np.savetxt(outpath+'/test/x.txt',tex,'%d');

    pass
train2600_path= oridata_path+'/train';
test2600_path = oridata_path+'/test';
def pca_show():
    x = np.loadtxt(train2600_path+'/x.txt',np.float);
    y = np.loadtxt(train2600_path+'/y.txt',np.int);
     
    idx = [];
    for i in range(4):
        idx.append(np.where(y==i)[0])
    
#     x = tranf2Guass(x, 1);
    x = tranf2One2(x,1)
    x = tranf2Guass(x,0)
    pca_y,_ = pca(x,dim=800);
     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d');
    cs = ['b','r','m','g'];

    for i in range(0,4):
        ax.scatter(pca_y[idx[i],0],pca_y[idx[i],1],pca_y[idx[i],2],c=cs[i]);
    plt.show();
    
        


def run():
    init_origin_data();
#     TSNE_dim_reduce();
    pca_show();
    pass; 
   
    
    

if __name__ == '__main__':
    run();
    pass