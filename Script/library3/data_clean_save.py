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
# In[5]:


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

human_contact_list  = ['sg_infectious_contact', 'primaryschool', 'highschool_2011', 'highschool_2012','highschool_2013', 'ht09_contact', 'SFHH', 'tij_lnVS', 'tij_lnVS2', 'haggle', 'Hospital']

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

def event_rate(period,df):
    df1 = df.copy()
    df1.timestamp = df1.timestamp%(period+1)
    strenght = float(df1.shape[0])
    period = float(period)
    return (period/strenght)*df1.groupby(df1.timestamp).size()



def remove_period(df1,plot=True):
    df = df1.copy()
    period = max(df.timestamp)
    ev_rate = event_rate(period,df)
    rescaled_time = np.cumsum(ev_rate)
    rescaled_time = rescaled_time.shift(1)
    rescaled_time[0] = 0
    print 'consistency' + str(int(max(rescaled_time+ev_rate.iloc[-1]))== period)
    print int(max(rescaled_time+ev_rate.iloc[-1])), period,sum(ev_rate)
    new_time = df.timestamp.apply(lambda x:rescaled_time[x%(period+1)])
    df_new = pd.DataFrame(new_time,columns=['timestamp'])
    
    if plot:
        plt.plot(df_new.groupby(df_new.timestamp).size())
        plt.show()
    df_new['count'] = 1
    print max(df_new.timestamp)+ev_rate.iloc[-1],max(df.timestamp) 
    if plot:
        plt.plot(df_new.groupby(df_new.timestamp).size())
        plt.plot(df.groupby(df.timestamp).size())
        plt.show()
        plt.plot(df_new.timestamp.unique(),(np.cumsum(event_rate(max(df.timestamp),df)).values/df_new.timestamp.unique()))
        plt.plot(df.timestamp.unique(),(np.cumsum(event_rate(max(df.timestamp),df)).values/df.timestamp.unique()))
        plt.show()
    df_new['nodes'] = df.nodes
    df_new.timestamp = df_new.timestamp - min(df_new.timestamp)
    return df_new









# In[6]:

def get_df_from_raw(title1):
    if title1 in ['tij_lnVS','ME','highschool_2013','EU','CollegeMsg','SFHH','tij_lnVS2']:delimiter = ' '
    if title1 in ['primaryschool','ht09_contact','haggle','DNC_Mail','highschool_2011','highschool_2012','Hospital']:delimiter = '\t'
    if title1 == 'DNC_Mail':header = 0
    else: header = -1
    
    ## if infectious
    if title1 == "sg_infectious_contact":
        df_list = glob.glob(path_dict[title1]+'*')
        i = 0
        for x in glob.glob(path_dict[title1]+'*'):
            df_list[i] = pd.read_table(x,header = -1)
            i+=1
        df = pd.concat(df_list)
    else: df = pd.read_csv(path_dict[title1],delimiter=delimiter,header=header)
    
    
    
    if title1 == 'DNC_Mail': df.reset_index(inplace = True)

    if title1 == 'ME':
        df.drop([2,3],axis=1,inplace=True)
        df.columns = ['start','stop','timestamp']
    elif title1 in ['DNC_Mail']:
        df.columns = ['start','stop','weights','timestamp']
        df.drop('weights',inplace=True,axis=1)
    elif title1 in ['tij_lnVS','tij_lnVS2','ht09_contact','sg_infectious_contact','SFHH']: df.columns = ['timestamp','start','stop']
    elif title1 in ['highschool_2013','highschool_2012','primaryschool','Hospital']: df.columns = ['timestamp','start','stop','comm1','comm2']
    else: df.columns = ['start','stop','timestamp']
    
    if title1 == 'tij_lnVS': 
        comm = pd.read_table(path+'Data/raw_data/metadata_InVS13.txt',header=-1)
        comm.columns = ['node','comm']
        comm = comm.set_index('node').to_dict()['comm']
        df['comm1'] = df.start.apply(lambda x:comm[x])
        df['comm2'] = df.stop.apply(lambda x:comm[x])
    df['nodes'] = df.apply(lambda x:tuple(sorted([x.start,x.stop])),axis=1).apply(lambda x:(int(x[0]),int(x[1])))
    df.start = df.nodes.apply(lambda x:x[0])
    df.stop = df.nodes.apply(lambda x:x[1])
    min1 = min(df.timestamp)
    df.timestamp = df.timestamp - min(df.timestamp)
    df = df.sort_values('timestamp',ascending=True).reset_index().drop('index',axis=1)
    df = df[df.start!=df.stop]
    df.head()
    return df


def get_df_connected_from_raw(title1,active_thresh = 0,remove_outlier = True):
    
    stop_dict = {}
    stop_dict["tij_lnVS"] = 9
    stop_dict["tij_lnVS2"] = 9
    
    stop_dict["Hospital"] = 1
    stop_dict["SFHH"] = 1

    
    stop_dict["ht09_contact"] = 3

    stop_dict["primaryschool"] = 1
    stop_dict["highschool_2013"] = 4
    stop_dict["highschool_2012"] = 6


    stop_dict["CollegeMsg"] = 1
    stop_dict["EU"] = 3
    
    
    ### obtain df from raw data
    df = get_df_from_raw(title1)
    
    ### create the time aggregated topology
    edge_weights = df.groupby(df.nodes).size()
    
    ### optional filter of low activity links
    edge_weights = edge_weights[edge_weights>= active_thresh]
    act_nodes = edge_weights.index
    df = df[df.nodes.isin(act_nodes)]
    
    
    ##### if computing the total number of days of DNC_mail before splitting in DNC_Mail2, comment this line!!!!!
    if title1 == 'DNC_Mail': df=df.iloc[1:]  
        
        
        
    edge_weights = pd.DataFrame(edge_weights).reset_index()
    edge_weights.columns = ['nodes','weights']
    edge_weights = edge_weights.apply(lambda x:(x.nodes[0],x.nodes[1],x.weights),axis=1).values

    g1 = nx.Graph()
    g1.add_weighted_edges_from(edge_weights)
    L = nx.line_graph(g1)
    print 
    giant = max(nx.connected_component_subgraphs(g1), key=len)
    connected_nodes = set(giant.nodes())
    print 'connected: '+ str(len(set(g1.nodes())))+', '+str(len(connected_nodes))
    df = df[(df.start.isin(connected_nodes))|(df.stop.isin(connected_nodes))]
    df = df[df.start != df.stop]
    new_index = pd.Series(index = sorted(list(set(df.start)|set(df.stop))),data = range(1,len(set(df.start)|set(df.stop))+1)).to_dict()
    df.start = df.start.apply(lambda x:new_index[x])
    df.stop = df.stop.apply(lambda x:new_index[x])
    df['nodes'] = df.apply(lambda x:tuple(sorted([int(x.start),int(x.stop)])),axis=1)
    df.timestamp = df.timestamp.astype('int')
    df.timestamp = df.timestamp - min(df.timestamp)
    df.reset_index(inplace = True)
    df.drop('index',axis = 1,inplace = True)
    df.drop_duplicates(['timestamp','nodes'],inplace = True)
    df.reset_index(inplace = True)
    df.drop('index',axis = 1,inplace = True)
    shift = df.timestamp.shift(-1) - df.timestamp
    
    

    if title1 in stop_dict.keys():
        stops = shift.sort_values(ascending=False)[:stop_dict[title1]]
        print stops
        if title1 in ['EU']:
            df = df[df.index<=stops.index[0]]
            stops = shift.sort_values(ascending=False)[1:stop_dict[title1]]
            
            
        if title1 in ['CollegeMsg']:
            df = df[df.index>stops.index[0]]
            stops = shift.sort_values(ascending=False)[1:stop_dict[title1]]
            if title1 == 'DNC_Mail': df.timestamp = df.timestamp - min(df.timestamp)
            print '22'
        print '3'
        if remove_outlier:
            for index in stops.index:
                print 'index: '+str(index)
                print 'shape: '+str(df.shape[0])
                df.loc[index+1:,'timestamp'] = df.loc[index+1:,'timestamp'].apply(lambda x: x-stops[index])
    df = df[['start','stop','timestamp','nodes']]
    df.drop_duplicates(['timestamp','nodes'],inplace = True)
    df = df[df.start!=df.stop]
    df.timestamp = df.timestamp - min(df.timestamp)
    return df



def compute_node_dist(node,g1):
    total_dic = {}
    for node1 in g1.nodes():
        if (node == node1) :
            total_dic[(node,node1)] = 0
        else:
            total_dic[(node,node1)] = nx.shortest_path_length(g1,node,node1,weight=None)
    return total_dic



def from_df_to_graphs(df):
    edge_weights = df.groupby('nodes').size()
    edge_weights = pd.DataFrame(edge_weights).reset_index()
    edge_weights.columns = ['nodes','weights']
    edge_weights = edge_weights.apply(lambda x:(x.nodes[0],x.nodes[1],x.weights),axis=1).values

    g1 = nx.Graph()
    g1.add_weighted_edges_from(edge_weights)
    L = nx.line_graph(g1)
    return g1,L



def from_graph_to_distance(g1):


    n = len([x for x in g1.nodes()])

    total_dic = [compute_node_dist(x,g1) for x in g1.nodes()]

    total_dic = {k: v for d in total_dic for k, v in d.items()}

    print "first consistency check: "+ str(len(total_dic)==n**2)
    return total_dic

def compute_line_g_dist_from_g(edge,g1,total_dic):
    edge_dic_distance = {}
    for edge1 in g1.edges():
        if (edge == edge1) :
            edge_dic_distance[(edge,edge1)] = 0
        else: 

            node_distance = min([total_dic[x] for x in itertools.product(list(edge), list(edge1))])+1
            edge_dic_distance[(edge,edge1)] = node_distance

    return edge_dic_distance


def from_line_graph_to_distance(g1,total_dic):
    E = len([x for x in g1.edges()])

    line_graph_distance_list = [compute_line_g_dist_from_g(x,g1,total_dic) for x in g1.edges()]

    line_graph_distance_voc= {k: v for d in line_graph_distance_list for k, v in d.items()}

    print "second consistency check: "+ str(len(line_graph_distance_voc)==E**2)
    
    return line_graph_distance_voc

def get_df_connected_from_raw_DNC(large_part = True,t_0 = None,active_thresh = 0):
    title1 = 'DNC_Mail'
    df = get_df_from_raw_DNC() 
    
    df['realtime'] = df.timestamp.apply(lambda x: pd.to_datetime(x,unit ='s'))
    t0_int = (pd.to_datetime(t_0) - min(df.realtime)).total_seconds() + min(df.timestamp)
    print t0_int
    df = df.iloc[1:]
    if large_part == True: df = df[df.realtime>=pd.to_datetime(t_0)]
    else:
        df = df[df.realtime<pd.to_datetime(t_0)]
        
    edge_weights = df.groupby(df.nodes).size()
    edge_weights = edge_weights[edge_weights>= active_thresh]
    act_nodes = edge_weights.index
    df = df[df.nodes.isin(act_nodes)]
    
    edge_weights = pd.DataFrame(edge_weights).reset_index()
    edge_weights.columns = ['nodes','weights']
    edge_weights = edge_weights.apply(lambda x:(x.nodes[0],x.nodes[1],x.weights),axis=1).values

    g1 = nx.Graph()
    g1.add_weighted_edges_from(edge_weights)
    L = nx.line_graph(g1)
    giant = max(nx.connected_component_subgraphs(g1), key=len)
    connected_nodes = set(giant.nodes())
    print len(g1.nodes),len(connected_nodes)
    df = df[(df.start.isin(connected_nodes))|(df.stop.isin(connected_nodes))]
    df = df[df.start != df.stop]
    new_index = pd.Series(index = sorted(list(set(df.start)|set(df.stop))),data = range(1,len(set(df.start)|set(df.stop))+1)).to_dict()
    df.start = df.start.apply(lambda x:new_index[x])
    df.stop = df.stop.apply(lambda x:new_index[x])
    df['nodes'] = df[['start','stop']].apply(lambda x:tuple(sorted([int(x.start),int(x.stop)])),axis=1)
    df.reset_index(inplace = True)
    df.drop('index',axis = 1,inplace = True)
    
    df.drop_duplicates(['timestamp','nodes'],inplace=True)
    
    if large_part == True: assert min(df.timestamp)>= t0_int
    else: 
        assert max(df.timestamp)< t0_int
    df.timestamp = df.timestamp - min(df.timestamp)
    return df


