# -*- coding: utf-8 -*-
'''
Created on 2018年12月1日

@author: zwp
'''

'''
尝试提取特征用GDBT,xgboost等方法

'''
import numpy as np;
import pandas as pd;
import time;
import h5py;
import os;
import xgboost as xgb
import tensorflow as tf;
from tensorflow import keras;
from tools import SysCheck
from tensorflow.keras.layers import Conv2D,AveragePooling2D,Input,Flatten,Dense,Dropout,MaxPool2D,concatenate;
from tensorflow.keras import Model
from tensorflow.keras.initializers import glorot_uniform
from xgboost import plot_importance
from matplotlib import pyplot as plt

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
result_path = base_path+'/Dataset/tianci/LCZ/result.csv'

out_path = base_path+'/Dataset/tianci/LCZ/round1_test_a_20181109.h5'

model_save_path = base_path+'/Dataset/tianci/LCZ';

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

rmv=0.5
def rmsame(X):
    def reg_f(x):
        v_mean = np.mean(x,axis=0,keepdims=True);
        v_std = np.std(x,axis=0,keepdims=True);
        v_rag = v_mean+rmv*v_std;
        v_rag = np.concatenate([v_rag,v_mean-rmv*v_std],axis=0);
        for i in range(x.shape[1]):
            tmp = x[:,i];
            idxu = np.where(tmp>v_rag[0,i])
            idxd = np.where(tmp<v_rag[1,i])
            tmp[idxu]=v_mean[0,i];
            tmp[idxd]=v_mean[0,i];
        return x;
    return np.array(list(map(reg_f,X)));


case=2;
val_case=1;
learn_rate=[0.005,0.005,0.005];
batch_size=30;
epoch=[50,10,10];
seed = 1212121;
input_shape=(1024,10,1)
load_model=True;


def get_featue(X,shuffle=False):
    def fet(x):
        if shuffle:
            np.random.shuffle(x);
        x = pd.DataFrame(x);
        # 最大值特征
        ret = x.max().values;
        # 最小值特征
        ret = np.append(ret,x.min().values);
        # 均值特征
        ret = np.append(ret,x.mean().values);
        # 标准差特征
        ret = np.append(ret,x.std().values);
        # 偏度特征
        ret = np.append(ret,x.skew().values);
        return ret;
    return np.array(list(map(fet,X)));
    
                
def run():

    train_path = base_path+'/Dataset/tianci/LCZ/splited/training%d.h5'%case
    test_paht = base_path+'/Dataset/tianci/LCZ/splited/test%d.h5'%case
    
#     train_path = base_path+'/Dataset/tianci/LCZ/training.h5'
    
    train_set = h5py.File(train_path);
    test_set = h5py.File(test_paht);
    out_set = h5py.File(out_path);
    
    train_sen1 = np.array(train_set['sen2']);
    train_label = np.array(train_set['label']);
    test_sen1 = np.array(test_set['sen2']);
    test_label = np.array(test_set['label']);

    out_sen  = np.array(out_set['sen2']);
    train_label = np.argmax(train_label,axis=1);
    test_label = np.argmax(test_label,axis=1);
    ################ 数据转换处理 #####################
    train_sen1 = newReg(train_sen1);
    test_sen1 = newReg(test_sen1);
#     out_sen = newReg(out_sen);
    
#     train_sen1 = regtoOne(train_sen1);
#     test_sen1 = regtoOne(test_sen1);    
    ## 更换展开方向
    train_sen1 = train_sen1.reshape((train_sen1.shape[0],
                            -1,train_sen1.shape[3])) 
    
    test_sen1 = test_sen1.reshape((test_sen1.shape[0],
                            -1,test_sen1.shape[3]))     
    out_sen = out_sen.reshape((out_sen.shape[0],-1,out_sen.shape[3]));
    
####################### 预处理 ######################
#     train_sen1 = rmsame(train_sen1);
#     test_sen1 = rmsame(test_sen1);
    
####################### 特征转化 ######################

    print('featue start')
    train_feat = get_featue(train_sen1, shuffle=False);
    test_feat =  get_featue(test_sen1, shuffle=False);
    print(train_feat);
    print('featue end')
    
######################### xgboost ##############


#     dtrain = xgb.DMatrix(train_feat,label=train_label);
#     dtest = xgb.DMatrix(test_feat,label=test_label);
#     # specify parameters via map
#     param = {
#             'booster': 'gbtree',
#             'objective': 'multi:softmax',  # 多分类的问题
#             'num_class': 17,               # 类别数，与 multisoftmax 并用
#             'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
# #             'max_depth': 12,               # 构建树的深度，越大越容易过拟合
# #             'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
# #             'subsample': 0.7,              # 随机采样训练样本
# #             'colsample_bytree': 0.7,       # 生成树时进行的列采样
# #             'min_child_weight': 3,
#             'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
#             'eta': 0.05,                  # 如同学习率
# #             'seed': 1000,
# #             'nthread': 4,                  # cpu 线程数
#             }
#     num_round = 50
#     bst = xgb.train(param, dtrain, num_round,evals=[(dtest,'t1')])
#     # make prediction
#     # ntree_limit must not be 0
#     preds = bst.predict(dtest, ntree_limit=num_round)
#     print(preds);
    
    
    param = {
            'booster': 'gbtree',
            'objective': 'multi:softmax',  # 多分类的问题
            'num_class': 17,               # 类别数，与 multisoftmax 并用
            'num_leaves':31,
            'n_estimatores':50,        # epoch
#             'min_data_in_leaf':31,
            'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth': 10,               # 构建树的深度，越大越容易过拟合
#             'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#             'subsample': 0.7,              # 随机采样训练样本
#             'colsample_bytree': 0.7,       # 生成树时进行的列采样
#             'min_child_weight': 3,
            'silent': 0,                   # 设置成1则没有运行信息输出，最好是设置为0.
            'learning_rate': 0.05,                  # 如同学习率
            'feature_fraction': 0.8,
            'bagging_fraction': 0.95,
            'bagging_freq': 2,
#             'metric': 'acc',
            'max_bin':128,
#             'seed': 1000,
#             'nthread': 4,                  # cpu 线程数
            }    
    
    
    clf = xgb.XGBClassifier(**param)

    clf.fit(train_feat, train_label,
            eval_set=[(test_feat, test_label)],
            eval_metric='mlogloss',
            verbose=True)
    
    evals_result = clf.evals_result()
    print(evals_result)
    
    
    preds = clf.predict(test_feat);
    
    
    
    
    py=preds;
    y = test_label;
    print(py);
    print(y);
    ss = len(py);
    es = np.count_nonzero(py-y);
    print(ss,es);
    print((ss-es)*1.0/ss);    
    plot_importance(bst);
    plt.show();
#     
#     res  = model.predict(out_sen);
    
    
#     py=np.argmax(res,axis=1);
#     res_out = np.zeros((len(py),17),np.int);
#     res_out[np.arange(len(py)),py]=1;
#     print(res_out);
#     np.savetxt(result_path,res_out,'%d', delimiter=',');


def test():
    a = np.random.uniform(0,0.5,size=(1,10,5))
    print(a);
    print(get_featue(a))
    
    pass;

if __name__ == '__main__':
    run();
#     test();
#     get_model_sp((1024,10,1));
    pass
