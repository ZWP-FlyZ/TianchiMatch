# -*- coding: utf-8 -*-
'''
Created on 2018年10月19日

@author: zwp12
'''


import numpy as np;
from tools import SysCheck;


TYPE_STAR = 0;
TYPE_GALAXY = 1;
TYPE_QSO = 2;
TYPE_UNKNOWN = 3;

TYPE_LIST=[TYPE_STAR,TYPE_GALAXY,TYPE_QSO,TYPE_UNKNOWN];
TYPE_NAME_LIST=['star','galaxy','qso','unknown'];

def t2ichange(dtype):
    if dtype == 'star':
        return TYPE_STAR;
    elif dtype == 'galaxy':
        return TYPE_GALAXY;
    elif dtype == 'qso':
        return TYPE_QSO;
    elif dtype=='unknown':
        return TYPE_UNKNOWN;
    else:
        return -1;

def i2tchange(itype):
    return TYPE_NAME_LIST[itype];

def init_dataset(csv_path_in,txt_path_out):
    '''
        将csv 的index文件 type字段转换为整型
        star   -> 0
        galaxy -> 1
        qso    -> 2
        unknown-> 3
    '''
    f_in = open(csv_path_in,'r');
    f_out= open(txt_path_out,'w');
    for line in f_in:
        idx,_type = line.strip().split(',');
        nty = str(t2ichange(_type));
        nli = idx + '\t'+nty+'\n';
        f_out.write(nli);
    f_in.close();
    f_out.close();
    print('init_dataset finished! outpaht=',txt_path_out);


def random_spilter(txt_in_path,txt_out_path,need_size):
    
    need_train,need_test=need_size;
    oridata = np.loadtxt(txt_in_path,np.int);
    print('load data finished!');
    
    idx_all = [];
    idx_test_all = [];
    for gty in TYPE_LIST:
        idx = np.where(oridata==gty)[0];
        #  处理训练集
        idx = np.random.choice(idx,
                               need_train[gty]+need_test[gty],
                               replace=False);
        idx_test_all.append(idx[:need_test[gty]]);
        idx_all.append(idx[need_test[gty]:]);
        print(TYPE_NAME_LIST[gty],'finished!');
    
    
    idx = np.concatenate(idx_all);
    np.random.shuffle(idx);
    sped= oridata[idx];
    np.savetxt(txt_out_path+'/train_data.txt',sped,'%d');
    
    idx = np.concatenate(idx_test_all);
    np.random.shuffle(idx);
    sped= oridata[idx];
    np.savetxt(txt_out_path+'/test_data.txt',sped,'%d');




def run():
    
#     init_dataset(origin_path,inited_ori_path);
    
    '''
    star有442969（91.55%）,
    galaxy有5231（1.08%）,
    qso有1363（0.28%）,
    unknown有34288（7.08%）
    '''
    
#     train_need={TYPE_STAR:6000,
#                 TYPE_GALAXY:3000,
#                 TYPE_QSO:1000,
#                 TYPE_UNKNOWN:3000};
#     
#     test_need={TYPE_STAR:500,
#                TYPE_GALAXY:500,
#                TYPE_QSO:300,
#                TYPE_UNKNOWN:200};
#     
#     random_spilter(inited_ori_path,base_path,
#                    (train_need,test_need));
    
    pass;




if __name__ == '__main__':
    run();
    pass