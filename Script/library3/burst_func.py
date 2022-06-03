#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import joblib
import glob
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import random
import itertools
import networkx as nx

def burst_train(inter_ev,dt):
    ev_distr = np.zeros(len(inter_ev))
    i = 0
    
    cnt = Counter()
    d = 0
    c = 1
    ev_distr[0] = d
    while i < len(inter_ev):
        
        
        if inter_ev[i]<= dt:
            
            ev_distr[i+1] = d
            c+=1
            
            i +=1
    
        else:
            
            cnt.update(Counter([c]))
            
            d +=1
            
            if i<len(inter_ev)-1:ev_distr[i+1] = d
            
            i+=1
            c = 1
            
            continue
    assert (sum([k*v for k,v in cnt.items()]) == np.sum([k*v for k,v in Counter(Counter(ev_distr).values()).items()]))
    return cnt,ev_distr


def get_burst_train(df,dt):
    
    df_rest = df.copy()
    int_ev_time = [x for x in (df_rest.timestamp.shift(-1) - df_rest.timestamp).values if ((x!= np.nan))]
    b_train = burst_train(int_ev_time,dt)[0]
    bursts = burst_train(int_ev_time,dt)[1]
    df['burst'] = bursts
    return Counter(b_train),df,

def event_rate(period,df):
    df1 = df.copy()
    df1.timestamp = df1.timestamp%(period+1)
    strenght = float(df1.shape[0])
    period = float(period)
    return (period/strenght)*df1.groupby(df1.timestamp).size()