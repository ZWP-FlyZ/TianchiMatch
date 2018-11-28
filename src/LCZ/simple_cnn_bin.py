# -*- coding: utf-8 -*-
'''
Created on 2018年11月26日

@author: zwp12
'''


import numpy as np;
import time;
import h5py;
import tensorflow as tf;
from tensorflow import keras;
from tools import SysCheck
from tensorflow.keras.layers import Conv2D,AveragePooling2D,Input,Flatten,Dense,Dropout,concatenate;
from tensorflow.keras import Model

'''

1.卷积核大小小于（5，5）可能会好一些
2.

'''



base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
result_path = base_path+'/Dataset/tianci/LCZ/result.csv'

out_path = base_path+'/Dataset/tianci/LCZ/round1_test_a_20181109.h5'

model_save_path = base_path+'/Dataset/tianci/LCZ';

case=2;
learn_rate=0.005;
batch_size=20;
epoch=2;



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

def get_model(input_shape):
    
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
    X = Dense(128,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(64,activation='relu')(X);
    X = Dropout(0.4)(X);
    X = Dense(32,activation='relu')(X);
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
    
    
    train_set = h5py.File(train_path);
    test_set = h5py.File(test_paht);
    out_set = h5py.File(out_path);
    
    train_sen1 = np.array(train_set['sen2']);
    train_label = np.array(train_set['label']);
    test_sen1 = np.array(test_set['sen2']);
    test_label = np.array(test_set['label']);

    out_sen  = np.array(out_set['sen2']);

#     train_sen1 = newReg(train_sen1);
#     test_sen1 = newReg(test_sen1);

    train_label = np.argmax(train_label,axis=1);
    idx = np.where(train_label>9);
    train_label[:]=0;
    train_label[idx]=1;
    test_label = np.argmax(test_label,axis=1);
    idx = np.where(test_label>9);
    test_label[:]=0;
    test_label[idx]=1;
    
    model = get_model2((32,32,10));    
     
    model.compile(optimizer=keras.optimizers.Adagrad(learn_rate), 
              loss=keras.losses.sparse_categorical_crossentropy, 
              metrics=['accuracy'])
     
    his = model.fit(train_sen1,train_label,
              batch_size=batch_size, epochs=epoch,
              validation_data = (test_sen1,test_label));
    print(his.history);
              
#     model = keras.models.load_model(model_save_path+'/all_model.h5');
    model.save(model_save_path+'/all_model.h5');
    
    res  = model.predict(out_sen);
    print(res);
    
#     py=np.argmax(res,axis=1);
#     res_out = np.zeros((len(py),17),np.int);
#     res_out[np.arange(len(py)),py]=1;
#     print(res_out);
#     np.savetxt(result_path,res_out,'%d', delimiter=',');


def test():
    
    input_shape=(32,32,10);
    train_set = np.random.uniform(size=(1000,32,32,10));
    test_set = np.random.randint(0,17,size=1000);
    conv_model = get_model(input_shape)
    conv_model.compile(optimizer=keras.optimizers.Adagrad(0.001), 
                  loss=keras.losses.sparse_categorical_crossentropy, 
                  metrics=['accuracy'])
        
    conv_model.fit(train_set,test_set,10,2);


if __name__ == '__main__':
    run();
    pass