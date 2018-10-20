# -*- coding: utf-8 -*-
'''
Created on 2018年10月19日

@author: zwp12
'''

'''
从压缩包中根据目录加载需要的数据
'''

import zipfile;
import numpy as np;
import os;

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from tools.Normal import *

class DataSource():
    index = None;#当前index数组
    label = None;# 对应的标签列表
    zip_data = None;# 加载的压缩文件
    preprocess=None;# 预处理函数
    def __init__(self,index_path,zip_data_path,p=None):
        idx = np.loadtxt(index_path,np.int);
        self.index=idx[:,0];
        self.label = idx[:,1];
        if p==None:
            self.preprocess=lambda x:x;
        else:
            self.preprocess=p;
        self.zip_data = self._loadzip(zip_data_path);
    

    def reloadIndex(self,index_path):
        idx = np.loadtxt(index_path,np.int);
        self.index=idx[:,0];
        self.label = idx[:,1];
    
    def size(self):
        return len(self.index);
    

    
    def getFeature(self,did):
        dstr = self.zip_data.read('%s.txt'%(did)).decode('utf-8');
        ori = np.fromstring(dstr,sep=',');
        aft = self.preprocess(ori);
        return aft;
    
    def getX(self,start,end):
        x = [];
        for idx in self.index[start:end]:
            x.append(self.getFeature(idx));
        return np.array(x)
    
    def getAllX(self):
        self.getX(0,self.size());
    
    
    def getY(self,start,end,one_hot=False):
        labs = self.label[start:end];
        if one_hot:
            return self._one_hot(labs)
        else:
            return labs;
    
    def getAllY(self,one_hot=False):
        return self.getY(0, self.size(), one_hot);
    
    def getXY(self,start,end,one_hot=False):
        x = self.getX(start, end);
        y = self.getY(start, end, one_hot);
        return x,y
    
    def getAllXY(self,one_hot=False):
        return self.getXY(0,self.size(),one_hot);
    
    
    def saveAllToArray(self,path):
        if not os.path.isdir(path):
            os.makedirs(path)
        x,y = self.getAllXY();
        np.savetxt(path+'/x.txt',x,'%.10f');
        np.savetxt(path+'/y.txt',y,'%d');
        print('DataSource->save data finished');
    
    def _one_hot(self,labs):
        ret =np.zeros([len(labs),4],np.int);
        idx = [i for i in range(len(labs))];
        ret[idx,labs]=1;
        return ret;
        
    def _loadzip(self,path):
        z = zipfile.ZipFile(path,'r');
        print('DataSource->load zip finished');
        return z;
    

def run():
    pass;
#     path = 'E:/work/Dataset/tianci/LAMOST2/train_data.txt';
#     testpath = 'E:/work/Dataset/tianci/LAMOST2/test_data.txt';
# 
#     datapath = 'E:/work/Dataset/tianci/LAMOST2/train_data.zip';
#     
# #     arrpath  = 'E:/work/Dataset/tianci/LAMOST2/data_2600'
# #     ds = DataSource(path,datapath);
# #     ds.saveAllToArray(arrpath+'/train');
# #     ds.reloadIndex(testpath)
# #     ds.saveAllToArray(arrpath+'/test');
#     
#     
#     def prc(x):
#         return (x-np.mean(x))/(np.max(x)-np.min(x));
# #         return (x-np.mean(x))/np.std(x);
#      
#     ds = DataSource(path,datapath);
#     x,y = ds.getAllXY();
# #     x = tranf2One(x,1);
#     x = tranf2Guass(x,1);
#     print(x);
#     print(y);
#     
#     
# 
#     idx = [];
#     for i in range(4):
#         idx.append(np.where(y==i)[0])
#      
# #     digits_proj = TSNE(n_components=3).fit_transform(x)
#     pca_y = PCA(n_components=100).fit_transform(x);
#      
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d');
# #     bx = fig.add_subplot(111, projection='3d');
#     cs = ['b','r','m','g'];
#     j=0;
#     for i in range(1,4):
#         ax.scatter(pca_y[idx[i],0],pca_y[idx[i],1],pca_y[idx[i],2],c=cs[i]);
#         j+=1;
# #     for i in idx:
# #         bx.scatter(digits_proj[i,0],digits_proj[i,1],digits_proj[i,2],c=cs[j])
# #         j+=1;
# #     ax.set_xlabel('X Label')
# #     ax.set_ylabel('Y Label')
# #     ax.set_zlabel('Z Label')
#     plt.show()

if __name__ == '__main__':
    run();
    pass