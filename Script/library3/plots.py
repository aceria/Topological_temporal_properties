#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib
import glob
from collections import Counter
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
import itertools
import networkx as nx
from statsmodels.stats.weightstats import DescrStatsW 
import os
# In[5]:

import sys
sys.path.insert(0, '/Users/albertoceria/Desktop/Temporal_Network/library/')
from utilities import *

human_contact_list =[ 'sg_infectious_contact', 'primaryschool', 'highschool_2012','highschool_2013', 'ht09_contact', 'SFHH', 'tij_lnVS', 'tij_lnVS2', 'Hospital']

mail_list = ['DNC_Mail_part2','ME', 'CollegeMsg', 'EU']

title_list_plot = mail_list + human_contact_list
name_paper = ['DNC_2*','ME*','CM*','EEU*','Infectious','PS','HS2012','HS2013','HT2009','SFHH','WP','WP2','Hospital']
title_to_paper_name = {k:v for k,v in zip(title_list_plot,name_paper)}
    


import seaborn as sns
color_list = cm.rainbow(np.linspace(0, 1, len(title_list_plot)))
color_dic = {k:v for k,v in zip(title_list_plot,color_list)}



def plot_coarse_grain_data(title1,folder):    
    file = joblib.load(glob.glob('/Users/albertoceria/Desktop/Temporal_Network/Cluster_Activity/'+folder+'/'+title1+'/data/'+title1+'t_coarse_grain_mean_mean_dist_full.joblib')[0])
    return pd.Series({k:v/file['avg_dist'][title1] for k,v in file['mean'][title1].items()})




def plot_coarse_grain_random_mean(title1,folder,rand):
    files = [joblib.load(x) for x in glob.glob('/Users/albertoceria/Desktop/Temporal_Network/Cluster_Activity/'+folder+'/'+title1+'/rand_long_full/'+title1+'t_coarse_grain_mean_mean*') if 'full_final' in x]
    return pd.concat([pd.Series({k:v/x['avg_dist'][title1+rand] for k,v in x['mean'][title1+rand].items()}) for x in files],axis=1).mean(axis=1)

def plot_coarse_grain_random_std(title1,folder,rand):
    files = [joblib.load(x) for x in glob.glob('/Users/albertoceria/Desktop/Temporal_Network/Cluster_Activity/'+folder+'/'+title1+'/rand_long_full/'+title1+'t_coarse_grain_mean_mean*') if 'full_final' in x]
    return pd.concat([pd.Series({k:v/x['avg_dist'][title1+rand] for k,v in x['mean'][title1+rand].items()}) for x in files],axis=1).std(axis=1)




import seaborn as sns

def plot_distance_distr(set1,save = False):
#     dict1 = joblib.load('/Users/alberto/Desktop/Temporal_Network/basic_statistics/node_distance_distr.joblib')
    dict2 = joblib.load('/Users/albertoceria/Desktop/Temporal_Network/basic_statistics/event_distance_distr.joblib')
    i = 0
    
    if set1 == 'human': 
        list1 = human_contact_list
        marker = 'o'
    if set1 == 'mail': 
        list1 = mail_list
        marker = '^'
    if set1 in title_list_plot:
        list1 = [set1] 
        if list1[0] in mail_list:
            marker = '^'
        else: marker = 'o'
    color_list = sns.color_palette(n_colors=len(list1))
    for title1 in list1:
        if title1 == 'mit':continue
        print title1
#         plt.plot(dict1[title1]/sum(dict1[title1]),color = color_list[i],marker = 'o',label = title1+'_event')
        plt.plot(dict2[title1]/sum(dict2[title1]),color = color_list[i],marker = marker,label = title_to_paper_name[title1]+'')
        i+=1
    plt.legend()     
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel(r'$Pr[\eta = j]}$',size = 15)
    plt.xlabel(r'$j$',size = 15)

    if save:plt.savefig('node_event_distance_distribution_'+set1+'.eps',format = 'eps',bbox_inches = 'tight')
    
    
    
def plot_spatial_to_temporal_analysis(list1,save = False,preproc = True):
    
    if preproc == False: str_prepr = '_no_preproc'
    elif preproc == 'full':str_prepr = '_full_prepr'
    else: str_prepr = ''
    try:os.mkdir('/Users/albertoceria/Desktop/Temporal_Network/plot_results/spatial_to_temporal'+str_prepr)
    except: 'Folder already existing'
    color_list = cm.rainbow(np.linspace(0, 1, len(list1)))
    color_dic = {k:v for k,v in zip(list1,color_list)}
    
    
    

    
    i = 0
    for title1 in list1:
        
        dict2 = joblib.load('/Users/albertoceria/Desktop/Temporal_Network/Other_Networks/spatial_to_temporal_analysis'+str_prepr+'/'+title1+'_spatial_to_temporal_res_cnt.joblib')['distance_distr']

        idx = np.where(dict2.values>=dict2.values[0])[0]
        

        if title1 == 'mit':continue
#     if title1 == 'haggle':continue
#     if title1 == 'ht09_contact':continue

            
#     if title1 == 'sg_infectious_contact':continue

        
        norm = joblib.load('/Users/albertoceria/Desktop/Temporal_Network/Other_Networks/spatial_to_temporal_analysis'+str_prepr+'/'+title1+'_spatial_to_temporal_res_cnt.joblib')['mean']
        
        data = pd.Series({k:v[0]/norm for k,v in joblib.load('/Users/albertoceria/Desktop/Temporal_Network/Other_Networks/spatial_to_temporal_analysis'+str_prepr+'/'+title1+'_spatial_to_temporal_distr_mean_var.joblib').items()})[idx]
        plt.plot(data,marker = 'o',label =r'$\mathcal{G}$',color = color_dic[title1])
        
        for n in ['1','2','3']:
            print norm
            mean_list = pd.concat([pd.Series({k:v[0]/norm
                      for k,v in w.items()}) for w in [joblib.load(x).to_dict() 
                                                       for x in [y for y in glob.glob('/Users/albertoceria/Desktop/Temporal_Network/Other_Networks/spatial_to_temporal_analysis'+str_prepr+'/'+title1+'_rand'+n+'/'+title1+'*')
                                                                 if 'mean_var_rand'+n in y]]],axis=1).mean(axis=1)[idx]

            std_list = pd.concat([pd.Series({k:v[0]/norm 
                      for k,v in w.items()}) for w in [joblib.load(x).to_dict() 
                                                       for x in [y for y in glob.glob('/Users/albertoceria/Desktop/Temporal_Network/Other_Networks/spatial_to_temporal_analysis'+str_prepr+'/'+title1+'_rand'+n+'/'+title1+'*')
                                                                 if 'mean_var_rand'+n in y]]],axis=1).std(axis=1)[idx]


            if n=='1':plt.errorbar(x=mean_list.index,y=mean_list.values,yerr=std_list.values,label =r'$\mathcal{G}^1$',linestyle = '--',color = color_dic[title1],fmt = 'o',capsize = 5)
            if n=='2':plt.errorbar(x=mean_list.index,y=mean_list.values,yerr=std_list.values,label =r'$\mathcal{G}^2$',linestyle = '--',color = color_dic[title1],fmt = 'o',capsize = 5)
            if n =='3': plt.errorbar(x=mean_list.index,y=mean_list.values,yerr=std_list.values,label =r'$\mathcal{G}^3$',linestyle = '-.',color = color_dic[title1],fmt = 'o',capsize = 5)
        plt.legend( prop={'size': 15})
        plt.title(title_to_paper_name[title1],size = 25)
        plt.xlabel(r'$j$',size = 25)
        plt.ylabel(r"$ \frac{E[\mathcal{T}(\ell,\ell')|\eta(\ell,\ell') = j]}{E[\mathcal{T}(\ell,\ell')]} $",size = 25)
        plt.xticks(np.arange(0, max(mean_list.index)+1),fontsize=20)
        plt.yticks(fontsize=20)
        i+=1
        if save:plt.savefig('/Users/albertoceria/Desktop/Temporal_Network/plot_results/spatial_to_temporal'+str_prepr+
                            '/spatial_to_temporal_'+title1+'.eps',format = 'eps',bbox_inches='tight')
        plt.show()
        
        
def time_degeneration(title1):  
    df = get_df(title1)
    df1 = df[['start','timestamp']]
    df2 = df[['stop','timestamp']]

    df1 = pd.DataFrame(df1.groupby(['start','timestamp']).size())
    df2 = pd.DataFrame(df2.groupby(['stop','timestamp']).size())

    df1.reset_index(inplace=True)
    df1.columns = ['user','timestamp','count']
    df2.reset_index(inplace=True)
    df2.columns = ['user','timestamp','count']

    df_tot = pd.concat([df1,df2])

    size = df_tot.groupby(['user','timestamp'])['count'].sum()

    return size


def get_simultaneous_events(title1):
    df = get_df(title1)
    size = df.groupby(df.timestamp).size()
    size = pd.Series(Counter(size.values))
    size = size/sum(size)
    return size


def from_cnt_to_bin(cnt,bins):
    cnt = pd.Series(cnt)
    max1 = max(cnt.keys())
    
    res = max1//bins
    hist = np.zeros(bins+1)
    res_list = np.array([res * (i+0.5)  for i in range(bins+1)])
    
    hist_index1 = 0
    debug_cnt = 0
    for x in cnt.index:
        
        key = x
        if hist_index1 == 0:
            if key<res:
                
                hist[hist_index1]+=cnt[key]
        
        
            else:
                for i in range(hist_index1,bins+1):
                    
                    if res*i<=key<res*(i+1):
                        hist[i]+=cnt[key]
                        hist_index1=i
                        
                        break
        else:
        
            for i in range(hist_index1,bins+1):
                if res*i<=key<res*(i+1):
                    hist[i]+=cnt[key]
                    hist_index1=i
                    break
        debug_cnt +=cnt[key]
    print np.sum(hist)==debug_cnt
    return res_list,hist

def get_bool_df(df,df_rand):
    bool_df = df_random['nodes'] == df['nodes']
    print df.shape[0],bool_df[bool_df==True].shape[0]

    
def clean_cnt(cnt):
    df = pd.DataFrame(cnt.keys(),cnt.values()).reset_index()
    df.columns = ['count','index']
    df = df[~df['index'].isna()].sort_values('index')
    return df.set_index('index')['count'].to_dict()


def plot_cnt(cnt,bins,range1,norm = True,clean_cnt1 = True,cumulative = False,binned = True,log_bin = False,shift_zero = False):
    if clean_cnt1 ==True: cnt = clean_cnt(cnt)
    keys = np.sort(np.array(cnt.keys()))
    values1 = np.array([cnt[k] for k in keys])
    if shift_zero:keys = keys +1
    if norm == True: 
        values1 = values1.astype('float')/float(np.sum(values1))
      
    if binned:
        if log_bin  == True:
            bins=np.logspace(np.log10(range1[0]),np.log10(range1[1]),bins)
            values1,bins = np.histogram(a=keys,weights=values1,bins = bins)
#             print 'set_xscale("log")!'
        else:values1,bins = np.histogram(a=keys,weights=values1,bins = bins,range= range1,)
        bins = (bins[:-1] + bins[1:])/2.
    else:bins = keys
    if cumulative: values1 = np.cumsum(values1)
   
    return bins,values1
