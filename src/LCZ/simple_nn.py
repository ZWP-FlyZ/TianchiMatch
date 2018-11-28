# -*- coding: utf-8 -*-
'''
Created on 2018年11月21日

@author: zwp12
'''

import numpy as np;
import time;
import h5py;
import tensorflow as tf;
from tensorflow import keras;
from tools import SysCheck
from sklearn.ensemble import AdaBoostClassifier
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier




base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
result_path = base_path+'/Dataset/tianci/LCZ/result.csv'

out_path = base_path+'/Dataset/tianci/LCZ/round1_test_a_20181109.h5'

class_rat = np.array([0.01438277,0.06933416,0.08994341,0.02455118,0.04680645, 
                    0.10015155,0.00927729,0.11160555,0.03855082,0.03392495,
                    0.12175409,0.02700033,0.02600989,0.1174262 ,0.0067884 ,0.02241419,
                    0.14007878])


case=2;
learn_rate=0.006;
batch_size=20;
epoch=3;


# def reg(x):
#     minarr = np.min(x,axis=(0,1));
#     maxarr = np.max(x,axis=(0,1));
#     a = (x-minarr);
#     b = (maxarr-minarr);
#     return np.divide(a,b,out=np.zeros_like(a),where=b!=0);


def reg(x):
    minarr = np.min(x);
    maxarr = np.max(x);
    a = (x-minarr);
    b = (maxarr-minarr);
    return np.divide(a,b,out=np.zeros_like(a),where=b!=0);

def regtoOne(X):
    return np.array(list(map(reg,X)));


def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32,10)),
        keras.layers.Dense(100,activation=tf.nn.relu),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64,activation=tf.nn.relu),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32,activation=tf.nn.relu),
        keras.layers.Dense(17,activation=tf.nn.softmax)
    ]);
    
    model.compile(optimizer=keras.optimizers.Adagrad(learn_rate), 
                  loss=keras.losses.sparse_categorical_crossentropy, 
                  metrics=['accuracy']);
    return model;
        
def get_model2():
    model = keras.Sequential([
        keras.layers.Dense(100,input_shape=(32*32*10),activation=tf.nn.relu),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64,activation=tf.nn.relu),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32,activation=tf.nn.relu),
        keras.layers.Dense(17,activation=tf.nn.softmax)
    ]);
    
    model.compile(optimizer=keras.optimizers.Adagrad(learn_rate), 
                  loss=keras.losses.sparse_categorical_crossentropy, 
                  metrics=['accuracy']);
    return model;

def flatten2(x):
    return x.reshape(-1,10240);



#################### 方法无效 #########################
def adaboost_run():
    train_path = base_path+'/Dataset/tianci/LCZ/splited/training%d.h5'%case
    test_paht = base_path+'/Dataset/tianci/LCZ/splited/test%d.h5'%case
    
    
    train_set = h5py.File(train_path);
    test_set = h5py.File(test_paht);
    out_set = h5py.File(out_path);
    
    train_sen1 = np.array(train_set['sen2']);
    train_label = np.array(train_set['label']);
    test_sen1 = np.array(test_set['sen2']);
    test_label = np.array(test_set['label']);

    out_sen  = np.array(out_set['sen2']);

    train_sen1 = regtoOne(train_sen1);
    test_sen1 = regtoOne(test_sen1);

    #############################
    
    train_sen1 = flatten2(train_sen1);
    test_sen1 = flatten2(test_sen1);
    out_sen = flatten2(out_sen);
    ###############################


    train_label = np.argmax(train_label,axis=1);
    test_label = np.argmax(test_label,axis=1);
    ann_model = KerasClassifier(build_fn= get_model2, epochs=10, batch_size=10);

    
    boosted_ann = AdaBoostClassifier(base_estimator = ann_model);
    
    boosted_ann.fit(train_sen1,train_label);
   
    
#     model.fit(train_sen1,train_label,
#               batch_size=batch_size, epochs=epoch,
#               validation_data = (test_sen1,test_label));
    
    res  = boosted_ann.predict(out_sen);
    print(res);
    
    py=np.argmax(res,axis=1);
    res_out = np.zeros((len(py),17),np.int);
    res_out[np.arange(len(py)),py]=1;
    print(res_out);
    np.savetxt(result_path,res_out,'%d', delimiter=',');


def simple_run():
    train_path = base_path+'/Dataset/tianci/LCZ/splited/training%d.h5'%case
    test_paht = base_path+'/Dataset/tianci/LCZ/splited/test%d.h5'%case
    
    
    train_set = h5py.File(train_path);
    test_set = h5py.File(test_paht);
    out_set = h5py.File(out_path);
    
    train_sen1 = np.array(train_set['sen2']);
    train_label = np.array(train_set['label']);
    test_sen1 = np.array(test_set['sen2']);
    test_label = np.array(test_set['label']);

    out_sen  = np.array(out_set['sen2']);

    train_sen1 = regtoOne(train_sen1);
    test_sen1 = regtoOne(test_sen1);

    train_label = np.argmax(train_label,axis=1);
    test_label = np.argmax(test_label,axis=1);

    model = get_model();    
    
    model.fit(train_sen1,train_label,
              batch_size=batch_size, epochs=epoch,
              validation_data = (test_sen1,test_label));
    
    res  = model.predict(out_sen);
    print(res);
    
    py=np.argmax(res,axis=1);
    res_out = np.zeros((len(py),17),np.int);
    res_out[np.arange(len(py)),py]=1;
    print(res_out);
    np.savetxt(result_path,res_out,'%d', delimiter=',');    
    
    
def test():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32,8)),
        keras.layers.Dense(160,activation=tf.nn.relu),
        keras.layers.Dense(80,activation=tf.nn.relu),
        keras.layers.Dense(40,activation=tf.nn.relu),
        keras.layers.Dense(17,activation=tf.nn.softmax)
    ]);
    
    model.compile(optimizer=keras.optimizers.Adagrad(0.001), 
                  loss=keras.losses.sparse_categorical_crossentropy, 
                  metrics=['accuracy']);
    
    
    train_set = np.random.normal(size=(1000,32,32,8));
    test_set = np.random.randint(0,17,size=1000);
#     test_set = np.zeros((1000,17));
#     test_set[np.arange(1000),test_idx]=1;
    
    
    print(train_set,train_set.shape);
    print(test_set,test_set.shape);
    
    model.fit(train_set,test_set,batch_size=10, epochs=10);
    

if __name__ == '__main__':
    
    simple_run();
    
    
    pass