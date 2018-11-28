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
from tensorflow.keras.layers import Conv2D,AveragePooling2D,Input,Flatten,Dense,Dropout,MaxPool2D;
from tensorflow.keras import Model

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

input_shape=(1024,10,1)
load_model=True;


def get_model3(input_shape):
    in_x  = Input(input_shape);
    X = Conv2D(64,kernel_size=(5,5),
            strides=(1,1),padding='same',activation='relu')(in_x);
#     X = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X);
    X = Conv2D(64,kernel_size=(4,4),
            strides=(1,1),padding='same',activation='relu')(X);
    X = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X);
    
    X = Conv2D(64,kernel_size=(3,3),
            strides=(1,1),padding='same',activation='relu')(X);
#     X = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X);    
    X = Conv2D(64,kernel_size=(2,2),
            strides=(1,1),padding='same',activation='relu')(X);
    X = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X);    
    

#     X = concatenate([X,X1,X3],axis=3);
    print(X);
    X = Flatten()(X);
#     X = Dense(96,activation='relu')(X);
#     X = Dropout(0.4)(X);
    X = Dense(64,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(16,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(2,activation='softmax')(X);
    model = Model(inputs=in_x,outputs=X);
    return model;    

def get_model2(input_shape):
    in_x  = Input(input_shape);
    X = Conv2D(64,kernel_size=(3,3),
            strides=(1,1),padding='same',activation='relu')(in_x);
    X = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X);
    X = Conv2D(64,kernel_size=(3,3),
            strides=(1,1),padding='same',activation='relu')(X);
    X = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X);
    X = Conv2D(64,kernel_size=(3,3),
            strides=(1,1),padding='same',activation='relu')(X);
    X = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X);    
    X = Conv2D(64,kernel_size=(3,3),
            strides=(1,1),padding='same',activation='relu')(X);
    X = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X);    
    
    ####################################################################
    X1 = Conv2D(32,kernel_size=(5,5),
            strides=(1,1),padding='same',activation='relu')(in_x);
    X1 = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X1);
    X1 = Conv2D(32,kernel_size=(5,5),
            strides=(1,1),padding='same',activation='relu')(X1);
    X1 = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X1);
    X1 = Conv2D(32,kernel_size=(5,5),
            strides=(1,1),padding='same',activation='relu')(X1);
    X1 = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X1);    
    X1 = Conv2D(32,kernel_size=(5,5),
            strides=(1,1),padding='same',activation='relu')(X1);
    X1 = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X1); 
    
#     ####################################################################
    X2 = Conv2D(32,kernel_size=(2,2),
            strides=(1,1),padding='same',activation='relu')(in_x);
    X2 = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X2);
    X2 = Conv2D(32,kernel_size=(2,2),
            strides=(1,1),padding='same',activation='relu')(X2);
    X2 = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X2);
    X2 = Conv2D(32,kernel_size=(2,2),
            strides=(1,1),padding='same',activation='relu')(X2);
    X2 = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X2);    
    X2 = Conv2D(32,kernel_size=(2,2),
            strides=(1,1),padding='same',activation='relu')(X2);
    X2 = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X2);     


#     ####################################################################
    X3 = Conv2D(32,kernel_size=(4,4),
            strides=(1,1),padding='same',activation='relu')(in_x);
    X3 = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X3);
    X3 = Conv2D(32,kernel_size=(4,4),
            strides=(1,1),padding='same',activation='relu')(X3);
    X3 = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X3);
    X3 = Conv2D(32,kernel_size=(4,4),
            strides=(1,1),padding='same',activation='relu')(X3);
    X3 = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X3);    
    X3 = Conv2D(32,kernel_size=(4,4),
            strides=(1,1),padding='same',activation='relu')(X3);
    X3 = AveragePooling2D(pool_size=(2,2),strides=(2,2))(X3);



#     X = concatenate([X,X1,X3],axis=3);
    print(X);
    X = Flatten()(X);
#     X = Dense(96,activation='relu')(X);
#     X = Dropout(0.4)(X);
    X = Dense(64,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(16,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(2,activation='softmax')(X);
    model = Model(inputs=in_x,outputs=X);
    return model;    


def build_model(input_shape):
    in_x  = Input(input_shape);
    X = Conv2D(16,kernel_size=(5,3),
            strides=(2,1),padding='same',activation='relu')(in_x);
    X = MaxPool2D(pool_size=(3,1),strides=(3,1))(X); 
    X = Conv2D(32,kernel_size=(5,3),
            strides=(2,1),padding='same',activation='relu')(X);    
    X = MaxPool2D(pool_size=(3,1),strides=(3,1))(X);
    
    
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
    X = Conv2D(16,kernel_size=(5,3),
            strides=(2,1),padding='same',activation='relu')(in_x);
    X = MaxPool2D(pool_size=(3,1),strides=(3,1))(X); 
    X = Conv2D(32,kernel_size=(5,3),
            strides=(2,1),padding='same',activation='relu')(X);    
    X = MaxPool2D(pool_size=(3,1),strides=(3,1))(X);
    
    
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
    
def get_model_sp(input_shape):
    in_x  = Input(input_shape);
    X = Conv2D(16,kernel_size=(3,3),
            strides=(3,1),padding='valid',activation='relu')(in_x);
    X = Conv2D(32,kernel_size=(3,3),
            strides=(3,1),padding='valid',activation='relu')(X);
    X = MaxPool2D(pool_size=(5,1),strides=(5,1))(X); 
    
#     X = AveragePooling2D(pool_size=(5,1),strides=(5,1))(X);
# 
#     X = Conv2D(32,kernel_size=(5,3),
#             strides=(4,1),padding='valid',activation='relu')(X);
#     X = Conv2D(32,kernel_size=(4,1),
#             strides=(2,1),padding='valid',activation='relu')(X);
            
#     X = AveragePooling2D(pool_size=(5,1),strides=(4,1))(X);
    
#     X = Conv2D(32,kernel_size=(5,5),
#             strides=(3,1),padding='valid',activation='relu')(X);
#     X = Conv2D(32,kernel_size=(5,5),
#             strides=(3,1),padding='valid',activation='relu')(X);
#     X = MaxPool2D(pool_size=(5,1),strides=(5,1))(X);    
    
    
    
    ####################################################################
 
#     X = concatenate([X,X1,X3],axis=3);
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
    
    ### 随机化位置
#     for i in range(len(train_sen1)):
#         np.random.shuffle(train_sen1[i]);
#     for i in range(len(test_sen1)):
#         np.random.shuffle(test_sen1[i]);    
    
    ### 窗口模糊化 
    win_size=16;
    step=8;
    start=0;end=0;i=0;
    train_tmp=[];
    test_tmp=[];
    while end<1024:
        start=i*step;
        end = min(win_size+start,32*32);
        i+=1;
        tag = np.mean(train_sen1[start:end,:],axis=0);
        train_tmp.append(tag);
        tag = np.mean(test_sen1[start:end,:],axis=0);
        test_tmp.append(tag);        
    aft_train_sen = np.array(train_tmp);     
    aft_test_sen = np.array(test_tmp);
    
    aft_train_sen = aft_train_sen.reshape((aft_train_sen.shape[0],
                            -1,aft_train_sen.shape[2],1));
    aft_test_sen = aft_test_sen.reshape((aft_test_sen.shape[0],
                            -1,aft_test_sen.shape[2],1));
    
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
        
        modelckpit = keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_acc',
                            save_best_only=True);
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