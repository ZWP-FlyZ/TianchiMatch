# -*- coding: utf-8 -*-
'''
Created on 2018年11月27日

@author: zwp12
'''

import numpy as np;
import time;
import h5py;
import os;
import tensorflow as tf;
from tensorflow import keras;
from tools import SysCheck
from tensorflow.keras.layers import Conv2D,AveragePooling2D,Input,Flatten,Dense,Dropout,concatenate;
from tensorflow.keras import Model

'''

先二分类，然后10分类和7分类

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
batch_size=20;
epoch=[10,10,10];

input_shape=(32,32,10)
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

def get_model_c2(input_shape,out_class=2):
    
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
    
    X = Flatten()(X);
    X = Dense(64,activation='relu')(X);
    X = Dropout(0.6)(X);
    X = Dense(32,activation='relu')(X);
    X = Dropout(0.6)(X);
    X = Dense(16,activation='relu')(X);
    X = Dropout(0.6)(X);
    X = Dense(out_class,activation='softmax')(X);
    model = Model(inputs=in_x,outputs=X);
    return model;

def get_model_c10(input_shape):
    
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
    
    X = Flatten()(X);
    X = Dense(64,activation='relu')(X);
    X = Dropout(0.1)(X);
    X = Dense(32,activation='relu')(X);
    X = Dropout(0.1)(X);
    X = Dense(3,activation='softmax')(X);
    model = Model(inputs=in_x,outputs=X);
    return model;

def get_model_c7(input_shape):
    
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
    
    X = Flatten()(X);
    X = Dense(64,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(32,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(7,activation='softmax')(X);
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
    
    train_path = base_path+'/Dataset/tianci/LCZ/training.h5'
    
    train_set = h5py.File(train_path);
    test_set = h5py.File(test_paht);
    out_set = h5py.File(out_path);
    
    train_sen1 = np.array(train_set['sen2'][:90000]);
    train_label = np.array(train_set['label'][:90000]);
    test_sen1 = np.array(test_set['sen2']);
    test_label = np.array(test_set['label']);

    out_sen  = np.array(out_set['sen2']);

#     train_sen1 = newReg(train_sen1);
#     test_sen1 = newReg(test_sen1);

    ear_stop = keras.callbacks.EarlyStopping(monitor='val_acc',
                        min_delta=0.01,patience=2);

    train_label = np.argmax(train_label,axis=1);
    test_label = np.argmax(test_label,axis=1);
    ############################ 处理获得二分类标签数据
    
    classfil_list=[];
    
    for i in range(17):
        print('class%d start'%i);
        arr = [i]
        idx = np.where(np.isin(train_label,arr))[0];
        nidx = np.where(np.logical_not(np.isin(train_label,arr)))[0];
        nidx = np.random.choice(nidx,len(idx));
        train_label_c2=np.zeros(2*len(idx));
        train_label_c2[:len(idx)]=1;
        train_sp = train_sen1[idx];
        train_sp = np.concatenate([train_sp,train_sen1[nidx]],axis=0)
        
        idx = np.where(np.isin(test_label,arr))[0];
        nidx = np.where(np.logical_not(np.isin(test_label,arr)))[0];
        nidx = np.random.choice(nidx,len(idx));
        test_label_c2=np.zeros(2*len(idx));
        test_label_c2[:len(idx)]=1;
        test_sp = test_sen1[idx];
        test_sp = np.concatenate([test_sp,test_sen1[nidx]],axis=0)
        
        ###################### 训练二分类器阶段 #########################
        c2_model = get_model_c2(input_shape);    
        c2_model.compile(optimizer=keras.optimizers.Adagrad(learn_rate[0]), 
                  loss=keras.losses.sparse_categorical_crossentropy, 
                  metrics=['accuracy'])
         
        his = c2_model.fit(train_sp,train_label_c2,
                  batch_size=batch_size, epochs=epoch[0],
                  validation_data = (test_sp,test_label_c2),
                  callbacks=[ear_stop]);
        print(his.history);
        c2_model.save(model_save_path+'/class%i_model.h5'%i);
        classfil_list.append(c2_model);
        res  = c2_model.predict(out_sen);
        print(res);
    
def validation_run():
    train_path = base_path+'/Dataset/tianci/LCZ/splited/training%d.h5'%val_case
    test_paht = base_path+'/Dataset/tianci/LCZ/splited/test%d.h5'%val_case
    
    train_set = h5py.File(test_paht);
    out_set = h5py.File(out_path);
    
    train_sen1 = np.array(train_set['sen2']);
    train_label = np.array(train_set['label']);

    out_sen  = np.array(out_set['sen2']);

#     train_sen1 = newReg(train_sen1);
#     test_sen1 = newReg(test_sen1);

    train_label = np.argmax(train_label,axis=1);  
    
    classif_list=[];   
    for i in range(17):
        m_p = model_save_path+'/class%i_model.h5'%i;
        classif_list.append(keras.models.load_model(m_p));
    print('模型加载完成');
    py=[];
    print(len(train_label))
    for i in range(len(train_label)):
        sli = train_sen1[i:i+1];
        tmp=np.zeros((17,2));
        for j,cls in enumerate(classif_list):
            tmp[j]=cls.predict(sli)[0];
        idx = np.argmax(tmp[:,1]);
        py.append(idx);
        if i%100==0:
            print(i);
    py=np.array(py);   
    y = train_label;
    print(py);
    print(y);
    ss = len(py);
    es = np.count_nonzero(py-y);
    print(ss,es);
    print((ss-es)*1.0/ss);
    
     
#     py=np.argmax(res,axis=1);
#     res_out = np.zeros((len(py),17),np.int);
#     res_out[np.arange(len(py)),py]=1;
#     print(res_out);
#     np.savetxt(result_path,res_out,'%d', delimiter=',');


def test():
    
    input_shape=(32,32,10);
    train_set = np.random.uniform(size=(1000,32,32,10));
    test_set = np.random.randint(0,17,size=1000);
    conv_model = get_model_c2(input_shape)
    conv_model.compile(optimizer=keras.optimizers.Adagrad(0.001), 
                  loss=keras.losses.sparse_categorical_crossentropy, 
                  metrics=['accuracy'])
        
    conv_model.fit(train_set,test_set,10,2);


if __name__ == '__main__':
    run();
    validation_run();
    pass