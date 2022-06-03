#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib
import glob
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import random
import itertools
import networkx as nx

human_contact_list =[ 'sg_infectious_contact', 'primaryschool', 'highschool_2012','highschool_2013', 'ht09_contact', 'SFHH', 'tij_lnVS', 'tij_lnVS2', 'Hospital']

mail_list = ['DNC_Mail_part2','ME', 'CollegeMsg', 'EU']

title_list_plot = mail_list + human_contact_list
name_paper = ['DNC_2*','ME*','CM*','EEU*','Infectious','PS','HS2012','HS2013','HT2009','SFHH','WP','WP2','Hospital']
title_to_paper_name = {k:v for k,v in zip(title_list_plot,name_paper)}
    




def shuffle_df(df,seed):
    random.seed(a = int(seed))                       #set the seed
    
    # get a timeseries per node
    nodes_timestamp = df.groupby('nodes')['timestamp'].apply(lambda x:list(x))
    random.shuffle(nodes_timestamp.values)   #shuffle timeseries
    
    
    df_shuff = pd.DataFrame(list(itertools.chain.from_iterable(list(pd.DataFrame(nodes_timestamp,columns = ['timestamp']).reset_index().apply(lambda x:list(itertools.product([x.nodes],x['timestamp'])),axis = 1).values))),columns=['nodes','timestamp']).sort_values('timestamp').reset_index().drop('index',axis=1)
    
    df_shuff['start'] = df_shuff['nodes'].apply(lambda x:x[0])
    df_shuff['stop'] = df_shuff['nodes'].apply(lambda x:x[1])
    df_shuff.sort_values('timestamp',inplace = True)
    return df_shuff


# In[4]:


### shuffle the time series and shift them with a random value
def get_random(value1,value2,p):
    if value1 == 0: return -1 * np.random.randint(0,value2)
    elif value2 == 0: return np.random.randint(0,value1)
    else:
        if random.random()<p: return np.random.randint(0,value1)
        else : return -1 * np.random.randint(0,value2)




def rand_df2(df,seed):
    
    random.seed(seed)
    
    df = df.copy()
    nodes_timestamp = df.groupby('nodes')['timestamp'].apply(lambda x:list(x))
    
    nodes_timestamp = df.groupby('nodes')['timestamp'].apply(lambda x:np.array(x))
    max_time = max(nodes_timestamp.apply(lambda x: max(x)))
    min_time = min(nodes_timestamp.apply(lambda x: min(x)))
    print min_time
    
    # periodic shuffle
    nodes_timestamp = nodes_timestamp.apply(lambda x: (x + random.randint(0,max_time))%max_time)
    
    
    
    random.shuffle(nodes_timestamp.values)   #shuffle timeseries
    
    df_shuff = pd.DataFrame(list(itertools.chain.from_iterable(list(pd.DataFrame(nodes_timestamp,columns = ['timestamp']).reset_index().apply(lambda x:list(itertools.product([x.nodes],x['timestamp'])),axis = 1).values))),columns=['nodes','timestamp']).sort_values('timestamp').reset_index().drop('index',axis=1)
    
    df_shuff['start'] = df_shuff['nodes'].apply(lambda x:x[0])
    df_shuff['stop'] = df_shuff['nodes'].apply(lambda x:x[1])
    
    return df_shuff


def shuffle_timestamp(df,ts,seed):
    df1 = df.copy()
    random.seed(seed)
    times = ts.index.unique()
    shape1 = ts.index.unique().shape[0]
    print random.randint(0,shape1)
    df1['timestamp'] = df1['timestamp'].apply(lambda x: times[random.randint(0,shape1-1)])
    df1.sort_values('timestamp',inplace = True)
    return df1

def permute_timestamps(df,seed):
    df1 = df.copy()
    random.seed(a = int(seed))
    random.shuffle(df1.nodes.values)
    df1.start = df1.nodes.apply(lambda x:x[0])
    df1.stop = df1.nodes.apply(lambda x:x[1])
    return df1

def random_df_same_weight(df,seed):
    cnt_df = pd.DataFrame(df.groupby('nodes').size())
    cnt_df.columns = ['cnt']
    cnt_df = cnt_df.reset_index()
    rand_dict = {}
    [rand_dict.update(dic) for dic in cnt_df.groupby('cnt')['nodes'].apply(lambda x: list(x)).apply(lambda x:dict(zip(x,(shuffle_list(x,10))))).values]
    print set(rand_dict.keys()) == set(rand_dict.values())
    df_rand = df.copy()
    df_rand['nodes'] = df_rand['nodes'].apply(lambda x:rand_dict[x])

    df_rand['start'] = df_rand.nodes.apply(lambda x:x[0])
    df_rand['stop'] = df_rand.nodes.apply(lambda x:x[1])
    return df_rand

def shuffle_list(list1,seed):
    list2 = list1[:]
    random.seed(a=seed)
    random.shuffle(list2)
    return list2
    