# -*- coding: utf-8 -*-
'''
Created on 2018年4月8日

@author: zwp
'''

'''
采用k-近邻方法计算

'''

import numpy as np;
import time;
from LAMOST_base_func  import DataSource;




def normalization(X):
    '''
    归一化函数 范围[0,1]
    '''
    rk = np.ndim(X);
    if rk == 1:
        X = np.reshape(X,[-1,X.shape[0]]);
    max_xs = np.reshape(np.max(X,axis=1),[-1,1]);
    min_xs = np.reshape(np.min(X,axis=1),[-1,1]);
    X = (X - min_xs) / (max_xs-min_xs);
    if rk == 1:
        X = X.flatten();
    return X;

def predict_one(X,Y,tx):
    '''
    预测函数 
    X:训练集特征矩阵[batch,feature_x]
    Y:训练集标签[batch,4]
    tx:测试集特征矩阵[feature_x]
    return:测试集预测结果[4]
    '''
    dis = dist(X,tx);
    sorted_k_index = np.argsort(dis)[0:k];
    ky = Y[sorted_k_index];
    ky = np.sum(ky,axis=0);
    pmax = np.max(ky);
    args = np.argwhere(ky==pmax);
#     arg_index = args[0,0];
    if np.alen(args) == 1:
        arg_index = args[0,0];
    else: 
        arg_index = 3;
    result = np.zeros([4]);
    result[arg_index]=1;
    return result;

def predict(X,Y,tX):
    '''
    预测函数 
    X:训练集特征矩阵[batch,feature_x]
    Y:训练集标签[batch,4]
    tX:测试集特征矩阵[batch,feature_x]
    return:测试集预测结果[batch,4]
    '''
    result=[];
    i = 0;
    now = time.time();
    for tx in tX:
        py = predict_one(X,Y,tx);
        result.append(py);
        i+=1;
        if i != 0 and i % 20 ==0:
            print('step=%d time=%.2f'%(i,time.time()-now));
            now = time.time();
    return np.array(result);

def predict_all(X,Y,tX):
    '''
    预测函数 
    X:训练集特征矩阵[batch,feature_x]
    Y:训练集标签[batch,4]
    tX:测试集特征矩阵[batch,feature_x]
    return:测试集预测结果[batch,4]
    '''

    now = time.time();
    result = [];
    start = 0;
    tdata_size = tX.shape[0];
    while start<tdata_size:
        end = min(start+batch,tdata_size);
        dis = dist2(X,tX[start:end]);
        sorted_k_index = np.argsort(dis,axis=1)[:,0:k];
        ky=Y[sorted_k_index];
        ky=np.sum(ky,axis=1);
        max_args = np.argmax(ky, axis=1);
        tbatch = max_args.shape[0];
        res = np.zeros([tbatch,4]);
        res[np.arange(tbatch),max_args]=1;
        result.extend(res);
        print('start=%d time=%.2f'%(start,time.time()-now));
        start+=batch;
        now = time.time();
    return np.array(result);



def evel(py,y):
    
    data_size = py.shape[0];
    
    mxpy=np.reshape(np.max(py,axis=1),(-1,1));
    py =(py/mxpy).astype(int);
    
    y_sum_types = np.sum(y,axis=0);
    py_sum_types= np.sum(py,axis=0);
    
    py_indexes = np.reshape(np.argmax(py, axis=1),(-1,1));
    y_indexes = np.reshape(np.argmax(y, axis=1),(-1,1));
    
    err=0;
    err_type=np.array([0,0,0,0],dtype=float);
    
    for i in range(data_size):
        yi = y_indexes[i];
        pyi = py_indexes[i];
        if yi != pyi:
            err+=1;
            err_type[yi]+=1;
    print('y=\t',y_sum_types);
    print('py=\t',py_sum_types);
    print('err=\t',err_type);
    tp = (y_sum_types-err_type);
    recall = np.divide(tp, y_sum_types, out=np.zeros_like(tp), where=y_sum_types!=0);
    prec = np.divide(tp, py_sum_types, out=np.zeros_like(tp), where=py_sum_types!=0);
    
    print('recall\t',recall);
    print('prec\t',prec);
    tmp1 = 2*recall*prec;
    tmp2 = recall+prec;
    macro_f1 = np.mean(np.divide(tmp1,tmp2,out=np.zeros_like(tmp1),where=tmp2!=0));
    print('all=%d true=%d err=%d pr=%.2f%%,macro_f1=%.3f'%(data_size,data_size-err,err,err*100.0/data_size,macro_f1));
    return py;


base_path=r'/home/zwp/work/Dataset/tianci/LAMOST';
train_data_index = base_path+r'/index_train.csv';
train_data_zip = base_path+r'/first_train_data_20180131.zip';

test_data_index = base_path+r'/index_test.csv';
test_data_zip = base_path+r'/first_train_data_20180131.zip';


result_test_index=base_path+r'/first_test_index_20180131.csv';
result_test_data_zip=base_path+r'/first_test_data_20180131.zip';
result_test_out_path=base_path+r'/test_result.csv';

# result_test_index=base_path+r'/first_rank_index_20180307.csv';
# result_test_data_zip=base_path+r'/first_rank_data_20180307.zip';
# result_test_out_path=base_path+r'/test_result_rank.csv';


need_train = True;

# 近邻数
k=20;

batch = 20;

# 距离计算个公式
def dist(X,x):
    '''
    X:[batch,feature_x]的训练集
    x:[feature_x]测试数据项
    return [batch] 距离值
    '''
    # 欧式距离法
    result = np.sqrt(np.sum((X-x)**2,axis=1));

    return result;

def dist2(X,tX):
    tX  = np.reshape(tX,[tX.shape[0],1,tX.shape[1]]);
    result = np.sqrt(np.sum((X-tX)**2,axis=2));
    return result;
    pass;



def run():
    print('\n加载训练数据')
    train_ds = DataSource.DataSource(train_data_index,train_data_zip);
    X,Y = train_ds.getAllData();
    # X = normalization(X);
    train_ds.reload_index(test_data_index);
    tX,tY = train_ds.getAllData();
    # tX = normalization(tX);
    print('\n开始预测');
    
    PY = predict(X,Y,tX);
    
    print('\n开始测试')
    
    evel(PY,tY);
    pass;



if __name__ == '__main__':
    run();
    pass