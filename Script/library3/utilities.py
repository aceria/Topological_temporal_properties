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
from statsmodels.stats.weightstats import DescrStatsW 
# In[5]:


path = '../'

human_contact_list  = ['sg_infectious_contact', 'primaryschool', 'highschool_2012','highschool_2013', 'ht09_contact', 'SFHH', 'tij_lnVS', 'tij_lnVS2', 'Hospital']

mail_list = ['DNC_Mail_part2','ME', 'CollegeMsg', 'EU']

title_list_plot = mail_list + human_contact_list
name_paper = ['DNC_2','ME','CM','EEU','PS','HS2013','HT2009','WP','Infectious','WP2','SFHH','Hospital','HS2012']
title_to_paper_name = {k:v for k,v in zip(title_list_plot,name_paper)}
    

path_dict = {}
path_dict['DNC_Mail'] = path+'Data/raw_data/dnc-temporalGraph/out.dnc-temporalGraph'
path_dict['ht09_contact'] = path+'Data/raw_data/ht09_contact_list.dat.gz'
path_dict['highschool_2013'] = path+'Data/raw_data/High-School_data_2013.csv.gz'
path_dict['primaryschool'] = path+'Data/raw_data/primaryschool.csv.gz'
path_dict['ME'] = path+'Data/raw_data/ME/out.radoslaw_email_email'
path_dict['sg_infectious_contact'] = path+'Data/raw_data/sg_infectious_contact_list/'
path_dict['EU'] = path+'Data/raw_data/email-Eu-core-temporal.txt'
path_dict['CollegeMsg'] = path+'Data/raw_data/CollegeMsg.txt.gz'
path_dict['tij_lnVS'] = path+'Data/raw_data/tij_InVS.dat'
path_dict['tij_lnVS2'] = path+'Data/raw_data/tij_InVS15.dat_.gz'
path_dict['SFHH'] = path+'Data/raw_data/tij_SFHH.dat_.gz'
path_dict['Hospital'] = path+'Data/raw_data/detailed_list_of_contacts_Hospital.dat_.gz'
path_dict['highschool_2012'] = path+'Data/raw_data/highschool_2012.csv.gz'



# In[ ]:

def to_log_norm(list1):
    list1 = np.array(list1).astype('float')
    return to_log(list1/sum(list1))



def to_log(x):
    return np.log10(np.array(x))



def get_dt(df):
    return min([x for x in (df.timestamp.shift(-1) - df.timestamp).values if x!=0 and x!=np.nan])


def get_graph(title1):
    path1 = path + 'Data/preprocessed/'+title1+'/'+title1+'_graph.joblib'
    return joblib.load(path1)


def get_linegraph(title1):
    path1 = path + 'Data/preprocessed/'+title1+'/'+title1+'_line_graph.joblib'
    return joblib.load(path1)


def get_df(title1):
    path1 = path + 'Data/preprocessed/'+title1+'/'+title1+'_df.gz'
    df = pd.read_csv(path1)
    df.nodes = df.nodes.apply(lambda x:eval(x))
    df.drop('Unnamed: 0',axis = 1,inplace = True)
    df.timestamp = df.timestamp - min(df.timestamp)
    df = df.sort_values('timestamp',ascending=True).reset_index().drop('index',axis=1)
    df.drop_duplicates(['nodes','timestamp'],inplace = True)
    df = df[['start','stop','timestamp','nodes']]
    return df




def get_distance_line_voc(title1):
    return joblib.load(path+'Data/preprocessed/'+title1+'/'+title1+'_line_graph_distance_voc.joblib')




def clean_counter(cnt1,n):
    cnt1[0] =  cnt1[0] - n
    for i in cnt1.keys():
        cnt1[i] = cnt1[i]/2.
    return cnt1


def get_mean_distance_events(dist_voc,df):
    size_df = df.groupby(df.nodes).size()
    size = size_df.to_dict()
    tot_zeros = df.groupby(df.nodes).size().apply(lambda x:(x*(x-1))/2)
    cnt_ev1 = Counter()
    for x,y in itertools.combinations(df.nodes.unique(),2):
        cnt_ev1.update({dist_voc[x,y]:size[x]*size[y]})
    cnt_ev1[0] = sum(tot_zeros)
    n = df.shape[0]
    print 'consistency_check1: '+str(sum(cnt_ev1.values()) == (n*(n-1))/2)
    return pd.Series(cnt_ev1)



def get_mean_distance(dist_voc,df):
    
    cnt_ev1 = Counter()
    for x,y in itertools.combinations(df.nodes.unique(),2):
        cnt_ev1.update({dist_voc[x,y]:1})

    n = df.nodes.unique().shape[0]
    print 'consistency_check1: '+str(sum(cnt_ev1.values()) == (n*(n-1))/2)
    return pd.Series(cnt_ev1)


def get_rel_dist(dist_voc,list1):
    cnt = Counter()
    size = Counter(list1)
    for x,y in itertools.combinations(set(list1),2):
        cnt.update({dist_voc[(x,y)]: size[x]*size[y]})
    cnt[0] = sum([(x*(x-1))/2 for x in size.values()])

    
    return cnt



def create_folder(path1,name_folder):
    try:os.mkdir(path+path1+name_folder+"/")
    except: print path+path1+name_folder+"/ already exist"
        
        
        
class power_law:
    import random

    def __init__(self,gamma,cutoff):
        self.gamma = gamma;
        self.cutoff = cutoff;
        self.distr = np.zeros(100)
        self.distr[1:] = 1./((np.arange(1,cutoff))**(gamma))/sum(1./((np.arange(1,cutoff))**(gamma)))
        self.cum_distr = np.cumsum(self.distr)
    def get_random_sample(self,size=1):
        if size == 1:
            rand = random.random()
            for i in range(1,self.cum_distr.shape[0]):
                if self.cum_distr[i-1]<rand<=self.cum_distr[i]: return i
        else:
            res_array = np.zeros(size)
            
            for r in range(size):
                rand = random.random()
                for i in range(1,self.cum_distr.shape[0]):
                    if self.cum_distr[i-1]<rand<=self.cum_distr[i]:
                            res_array[r] = i
            return res_array
