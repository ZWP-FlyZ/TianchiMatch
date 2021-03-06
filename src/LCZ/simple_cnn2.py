# -*- coding: utf-8 -*-
'''
Created on 2018年11月28日

@author: zwp12
'''




import numpy as np;
import time;
import h5py;
import os;
import tensorflow as tf;
from tensorflow import keras;
from tools import SysCheck
from tensorflow.keras.layers import Conv2D,AveragePooling2D,Input,Flatten,Dense,Dropout,MaxPool2D,concatenate;
from tensorflow.keras import Model
from tensorflow.keras.initializers import glorot_uniform
'''

改变展开方向后cnn

'''

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
result_path = base_path+'/Dataset/tianci/LCZ/result.csv'

out_path = base_path+'/Dataset/tianci/LCZ/round1_test_a_20181109.h5'

model_save_path = base_path+'/Dataset/tianci/LCZ';

case=2;
val_case=1;
learn_rate=[0.005,0.005,0.005];
batch_size=30;
epoch=[50,10,10];
seed = 1212121;
input_shape=(1024,10,1)
load_model=True;

def build_model(input_shape):
    in_x  = Input(input_shape);
    X = Conv2D(8,kernel_size=(3,3),
            strides=(1,1),padding='same',activation='relu',
        kernel_initializer=glorot_uniform(seed=seed))(in_x);
    X = AveragePooling2D(pool_size=(2,1),strides=(2,1))(X); 
    X = Conv2D(16,kernel_size=(3,3),
            strides=(1,1),padding='same',activation='relu',
            kernel_initializer=glorot_uniform(seed=seed))(X);    
    X = AveragePooling2D(pool_size=(2,1),strides=(2,1))(X);
    
    X = Conv2D(32,kernel_size=(3,3),
            strides=(1,1),padding='same',activation='relu',
            kernel_initializer=glorot_uniform(seed=seed))(X);    
    X = AveragePooling2D(pool_size=(2,1),strides=(2,1))(X);    
    
    print(X);
    X = Flatten()(X);
    X = Dense(256,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(128,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(64,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(17,activation='softmax')(X);
    model = Model(inputs=in_x,outputs=X);
    return model;    


def get_model(input_shape):
    in_x  = Input(input_shape);
    X = Conv2D(32,kernel_size=(3,3),
            strides=(1,1),padding='same',activation='relu',
        kernel_initializer=glorot_uniform(seed=seed))(in_x);
    X = MaxPool2D(pool_size=(10,2),strides=(10,2))(X);
    X = Conv2D(32,kernel_size=(3,3),
            strides=(1,1),padding='same',activation='relu',
        kernel_initializer=glorot_uniform(seed=seed))(X);
    X = MaxPool2D(pool_size=(10,2),strides=(10,2))(X);

    X1 = Conv2D(32,kernel_size=(5,5),
            strides=(1,1),padding='same',activation='relu',
            kernel_initializer=glorot_uniform(seed=seed))(in_x);    
    X1 = MaxPool2D(pool_size=(10,2),strides=(10,2))(X1);
    X1 = Conv2D(32,kernel_size=(5,5),
            strides=(1,1),padding='same',activation='relu',
            kernel_initializer=glorot_uniform(seed=seed))(X1);    
    X1 = MaxPool2D(pool_size=(10,2),strides=(10,2))(X1);


    X2 = Conv2D(32,kernel_size=(7,7),
            strides=(1,1),padding='same',activation='relu',
            kernel_initializer=glorot_uniform(seed=seed))(in_x);    
    X2 = MaxPool2D(pool_size=(10,2),strides=(10,2))(X2);
    X2 = Conv2D(32,kernel_size=(7,7),
            strides=(1,1),padding='same',activation='relu',
            kernel_initializer=glorot_uniform(seed=seed))(X2);    
    X2 = MaxPool2D(pool_size=(10,2),strides=(10,2))(X2);
    X = concatenate([X,X1],axis=3);
    print(X);
    X = Flatten()(X);
    X = Dense(256,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(128,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(17,activation='softmax')(X);
    model = Model(inputs=in_x,outputs=X);
    return model;    
   
    
def get_model_sp(input_shape):
    in_x  = Input(input_shape);
    X = Conv2D(32,kernel_size=(3,3),
            strides=(1,1),padding='same',activation='relu',
        kernel_initializer=glorot_uniform(seed=seed))(in_x);
    X = MaxPool2D(pool_size=(10,2),strides=(10,2))(X);
    X = Conv2D(32,kernel_size=(3,3),
            strides=(1,1),padding='same',activation='relu',
        kernel_initializer=glorot_uniform(seed=seed))(X);
    X = MaxPool2D(pool_size=(10,2),strides=(10,2))(X);

     
    X1 = Conv2D(32,kernel_size=(5,5),
            strides=(1,1),padding='same',activation='relu',
            kernel_initializer=glorot_uniform(seed=seed))(in_x);    
    X1 = MaxPool2D(pool_size=(10,2),strides=(10,2))(X1);
    X1 = Conv2D(32,kernel_size=(5,5),
            strides=(1,1),padding='same',activation='relu',
            kernel_initializer=glorot_uniform(seed=seed))(X1);    
    X1 = MaxPool2D(pool_size=(10,2),strides=(10,2))(X1);


    X2 = Conv2D(32,kernel_size=(7,7),
            strides=(1,1),padding='same',activation='relu',
            kernel_initializer=glorot_uniform(seed=seed))(in_x);    
    X2 = MaxPool2D(pool_size=(10,2),strides=(10,2))(X2);
    X2 = Conv2D(32,kernel_size=(7,7),
            strides=(1,1),padding='same',activation='relu',
            kernel_initializer=glorot_uniform(seed=seed))(X2);    
    X2 = MaxPool2D(pool_size=(10,2),strides=(10,2))(X2);
    X = concatenate([X,X1],axis=3);
    print(X);
    X = Flatten()(X);
    X = Dense(256,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(128,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(17,activation='softmax')(X);
    model = Model(inputs=in_x,outputs=X);
    return model; 

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
    out_sen = newReg(out_sen);
    
#     train_sen1 = regtoOne(train_sen1);
#     test_sen1 = regtoOne(test_sen1);    
    ## 更换展开方向
    train_sen1 = train_sen1.reshape((train_sen1.shape[0],
                            -1,train_sen1.shape[3])) 
    
    test_sen1 = test_sen1.reshape((test_sen1.shape[0],
                            -1,test_sen1.shape[3]))     
    
    train_sen1 = rmsame(train_sen1);
    test_sen1 = rmsame(test_sen1);
    
    
    ### 随机化位置
#     for i in range(len(train_sen1)):
#         np.random.shuffle(train_sen1[i]);
#     for i in range(len(test_sen1)):
#         np.random.shuffle(test_sen1[i]);    
        
    train_sen1 = train_sen1.reshape((train_sen1.shape[0],
                            -1,train_sen1.shape[2],1));
    test_sen1 = test_sen1.reshape((test_sen1.shape[0],
                            -1,test_sen1.shape[2],1));    
    out_sen = out_sen.reshape((out_sen.shape[0],-1,out_sen.shape[3],1));
    
    if False:
        c2_model = keras.models.load_model(model_save_path+'/class127_model.h5');
    else:
        c2_model = get_model(train_sen1.shape[1:]);
        
    if True:    
        c2_model.compile(optimizer=keras.optimizers.Adagrad(learn_rate[0]), 
                  loss=keras.losses.sparse_categorical_crossentropy, 
                  metrics=['accuracy'])
        
        modelckpit = keras.callbacks.ModelCheckpoint(model_save_path+'/ckpt/ckpt{epoch:02d}-{val_acc:.2f}.h5', 
                                                     monitor='val_acc',
                            save_best_only=False);
        ear_stop = keras.callbacks.EarlyStopping(monitor='val_acc',
                            min_delta=0.005,patience=8); 
        his = c2_model.fit(train_sen1,train_label,
                  batch_size=batch_size, epochs=epoch[0],
                  validation_data = (test_sen1,test_label),
                  callbacks=[ear_stop,modelckpit]);
        print(his.history);
        c2_model.save(model_save_path+'/class127_model.h5');
    his = c2_model.evaluate(test_sen1, test_label);
    print(his);
    res  = c2_model.predict(out_sen);
    print(res);
    
    py=np.argmax(res,axis=1);
    res_out = np.zeros((len(py),17),np.int);
    res_out[np.arange(len(py)),py]=1;
    print(res_out);
    np.savetxt(result_path,res_out,'%d', delimiter=',');

if __name__ == '__main__':
    run();
#     get_model_sp((1024,10,1));
    pass