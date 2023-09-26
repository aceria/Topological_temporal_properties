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

# Define the path to the data directory
path = '../'

# Define lists of datasets for human contact and email communication
human_contact_list = ['sg_infectious_contact', 'primaryschool', 'highschool_2011', 'highschool_2012',
                      'highschool_2013', 'ht09_contact', 'SFHH', 'tij_lnVS', 'tij_lnVS2', 'haggle', 'Hospital']

mail_list = ['DNC_Mail_part2', 'ME', 'CollegeMsg', 'EU']

# Combine both lists for plotting purposes
title_list_plot = mail_list + human_contact_list

# Define paper names associated with the datasets
name_paper = ['DNC_2', 'ME', 'CM', 'EEU', 'PS', 'HS2013', 'HT2009', 'WP', 'Infectious', 'WP2', 'SFHH', 'Hospital', 'HS2012']

# Create a mapping of dataset titles to paper names
title_to_paper_name = {k: v for k, v in zip(title_list_plot, name_paper)}

# Define a dictionary mapping dataset titles to their respective file paths
path_dict = {
    'DNC_Mail': path + 'Data/raw_data/dnc-temporalGraph/out.dnc-temporalGraph',
    'ht09_contact': path + 'Data/raw_data/ht09_contact_list.dat.gz',
    'highschool_2013': path + 'Data/raw_data/High-School_data_2013.csv.gz',
    'primaryschool': path + 'Data/raw_data/primaryschool.csv.gz',
    'ME': path + 'Data/raw_data/ME/out.radoslaw_email_email',
    'sg_infectious_contact': path + 'Data/raw_data/sg_infectious_contact_list/',
    'EU': path + 'Data/raw_data/email-Eu-core-temporal.txt',
    'CollegeMsg': path + 'Data/raw_data/CollegeMsg.txt.gz',
    'tij_lnVS': path + 'Data/raw_data/tij_InVS.dat',
    'tij_lnVS2': path + 'Data/raw_data/tij_InVS15.dat_.gz',
    'SFHH': path + 'Data/raw_data/tij_SFHH.dat_.gz',
    'Hospital': path + 'Data/raw_data/detailed_list_of_contacts_Hospital.dat_.gz',
    'highschool_2012': path + 'Data/raw_data/highschool_2012.csv.gz'
}




# In[6]:

def get_df_from_raw(title1):
    """
    Read and process data from the specified dataset.

    Args:
        title1 (str): The title of the dataset to read.

    Returns:
        pd.DataFrame: Processed dataframe containing the dataset.
    """
    # Define delimiters and header based on the dataset title
    if title1 in ['tij_lnVS', 'ME', 'highschool_2013', 'EU', 'CollegeMsg', 'SFHH', 'tij_lnVS2']:
        delimiter = ' '
    if title1 in ['primaryschool', 'ht09_contact', 'haggle', 'DNC_Mail', 'highschool_2011', 'highschool_2012', 'Hospital']:
        delimiter = '\t'
    if title1 == 'DNC_Mail':
        header = 0
    else:
        header = -1

    # Read the dataset based on title and format
    if title1 == "sg_infectious_contact":
        # For infectious contact, read multiple files and concatenate them
        df_list = glob.glob(path_dict[title1] + '*')
        i = 0
        for x in glob.glob(path_dict[title1] + '*'):
            df_list[i] = pd.read_table(x, header=-1)
            i += 1
        df = pd.concat(df_list)
    else:
        # For other datasets, read using specified delimiter and header
        df = pd.read_csv(path_dict[title1], delimiter=delimiter, header=header)

    if title1 == 'DNC_Mail':
        df.reset_index(inplace=True)

    if title1 == 'ME':
        df.drop([2, 3], axis=1, inplace=True)
        df.columns = ['start', 'stop', 'timestamp']
    elif title1 in ['DNC_Mail']:
        df.columns = ['start', 'stop', 'weights', 'timestamp']
        df.drop('weights', inplace=True, axis=1)
    elif title1 in ['tij_lnVS', 'tij_lnVS2', 'ht09_contact', 'sg_infectious_contact', 'SFHH']:
        df.columns = ['timestamp', 'start', 'stop']
    elif title1 in ['highschool_2013', 'highschool_2012', 'primaryschool', 'Hospital']:
        df.columns = ['timestamp', 'start', 'stop', 'comm1', 'comm2']
    else:
        df.columns = ['start', 'stop', 'timestamp']

    if title1 == 'tij_lnVS':
        # Read community information for tij_lnVS dataset
        comm = pd.read_table(path + 'Data/raw_data/metadata_InVS13.txt', header=-1)
        comm.columns = ['node', 'comm']
        comm = comm.set_index('node').to_dict()['comm']
        df['comm1'] = df.start.apply(lambda x: comm[x])
        df['comm2'] = df.stop.apply(lambda x: comm[x])

    # Process and format node pairs
    df['nodes'] = df.apply(lambda x: tuple(sorted([x.start, x.stop])), axis=1).apply(lambda x: (int(x[0]), int(x[1])))
    df.start = df.nodes.apply(lambda x: x[0])
    df.stop = df.nodes.apply(lambda x: x[1])
    min1 = min(df.timestamp)
    if 'DNC_Mail' not in title1:df.timestamp = df.timestamp - min(df.timestamp)
    df = df.sort_values('timestamp', ascending=True).reset_index().drop('index', axis=1)
    df = df[df.start != df.stop]

    return df



def get_df_connected_from_raw(title1, active_thresh=0, remove_outlier=True,large_part = False,t_0 = None):
    """
    Process and filter data from the specified dataset, creating a temporal network dataframe.

    Args:
        title1 (str): The title of the dataset to process.
        active_thresh (int, optional): The threshold for filtering low-activity links. Defaults to 0.
        remove_outlier (bool, optional): Whether to remove outliers in inter-event times by shifting data. Defaults to True.
        large_part (bool,optional): Wheather to remove the part of DNC dataset with smallest number of nodes
        t_0 (int, opt): reference time to divide the DNC dataset (it is considered only for DNC dataset)
    Returns:
        pd.DataFrame: Processed dataframe containing the filtered temporal network.
    """
    
    if title1 == 'DNC_Mail':
        return get_df_connected_from_raw_DNC(True,t_0,active_thresh)
    
    else:
        # Define a dictionary of top inter-event time outlier to be removed for different datasets
        stop_dict = {
            "tij_lnVS": 9,
            "tij_lnVS2": 9,
            "Hospital": 1,
            "SFHH": 1,
            "ht09_contact": 3,
            "primaryschool": 1,
            "highschool_2013": 4,
            "highschool_2012": 6,
            "CollegeMsg": 1,
            "EU": 3
        }

        # Obtain the dataframe from raw data
        df = get_df_from_raw(title1)

        # Create the time-aggregated topology
        edge_weights = df.groupby(df.nodes).size()

        # Optional filter of low-activity links based on the specified threshold
        edge_weights = edge_weights[edge_weights >= active_thresh]
        act_nodes = edge_weights.index
        df = df[df.nodes.isin(act_nodes)]

        # If the dataset is 'DNC_Mail', skip the first row (commented line)
        if title1 == 'DNC_Mail':
            df = df.iloc[1:]

        # Resetting index and formatting data
        edge_weights = pd.DataFrame(edge_weights).reset_index()
        edge_weights.columns = ['nodes', 'weights']
        edge_weights = edge_weights.apply(lambda x: (x.nodes[0], x.nodes[1], x.weights), axis=1).values

        # Create a graph and its line graph
        g1 = nx.Graph()
        g1.add_weighted_edges_from(edge_weights)
        L = nx.line_graph(g1)

        # Finding the largest connected component
        giant = max(nx.connected_component_subgraphs(g1), key=len)
        connected_nodes = set(giant.nodes())

        # Filter data to include only nodes within the largest connected component
        df = df[(df.start.isin(connected_nodes)) | (df.stop.isin(connected_nodes))]
        df = df[df.start != df.stop]

        # Create a new index for nodes and update node pairs
        new_index = pd.Series(index=sorted(list(set(df.start) | set(df.stop))), data=range(1, len(set(df.start) | set(df.stop)) + 1)).to_dict()
        df.start = df.start.apply(lambda x: new_index[x])
        df.stop = df.stop.apply(lambda x: new_index[x])
        df['nodes'] = df.apply(lambda x: tuple(sorted([int(x.start), int(x.stop)])), axis=1)
        df.timestamp = df.timestamp.astype('int')
        df.timestamp = df.timestamp - min(df.timestamp)
        df.reset_index(inplace=True)
        df.drop('index', axis=1, inplace=True)

        # Drop duplicate entries based on timestamp and node pairs
        df.drop_duplicates(['timestamp', 'nodes'], inplace=True)
        df.reset_index(inplace=True)
        df.drop('index', axis=1, inplace=True)
        
        # Calculate the time shifts and remove inter-event times outliers based on stop values
        shift = df.timestamp.shift(-1) - df.timestamp

        if title1 in stop_dict.keys():
            stops = shift.sort_values(ascending=False)[:stop_dict[title1]]

            if title1 in ['EU']:
                df = df[df.index <= stops.index[0]]
                stops = shift.sort_values(ascending=False)[1:stop_dict[title1]]

            if title1 in ['CollegeMsg']:
                df = df[df.index > stops.index[0]]
                stops = shift.sort_values(ascending=False)[1:stop_dict[title1]]
                if title1 == 'DNC_Mail':
                    df.timestamp = df.timestamp - min(df.timestamp)

            if remove_outlier:
                for index in stops.index:
                    df.loc[index + 1:, 'timestamp'] = df.loc[index + 1:, 'timestamp'].apply(lambda x: x - stops[index])

        # Return the processed dataframe
        df = df[['start', 'stop', 'timestamp', 'nodes']]
        df.drop_duplicates(['timestamp', 'nodes'], inplace=True)
        df = df[df.start != df.stop]
        df.timestamp = df.timestamp - min(df.timestamp)

    return df


def compute_node_dist(node, g1):
    """
    Compute the shortest path distances from a given node to all other nodes in the graph.

    Args:
        node: The node for which distances are calculated.
        g1: The input graph.

    Returns:
        dict: A dictionary containing node pairs as keys and their corresponding shortest path distances as values.
    """
    total_dic = {}
    for node1 in g1.nodes():
        if node == node1:
            total_dic[(node, node1)] = 0
        else:
            # Calculate the shortest path length using networkx's built-in function.
            total_dic[(node, node1)] = nx.shortest_path_length(g1, node, node1, weight=None)
    return total_dic


def from_df_to_graphs(df):
    """
    Convert a dataframe to a graph representation.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        g1: The graph representation of the dataframe.
        L: The line graph representation of the dataframe.
    """
    edge_weights = df.groupby('nodes').size()
    edge_weights = pd.DataFrame(edge_weights).reset_index()
    edge_weights.columns = ['nodes', 'weights']
    edge_weights = edge_weights.apply(lambda x: (x.nodes[0], x.nodes[1], x.weights), axis=1).values

    # Create an undirected graph (g1) and its line graph (L) from edge weights.
    g1 = nx.Graph()
    g1.add_weighted_edges_from(edge_weights)
    L = nx.line_graph(g1)
    return g1, L


def from_graph_to_distance(g1):
    """
    Calculate the shortest path distances between all pairs of nodes in a graph.

    Args:
        g1: The input graph.

    Returns:
        dict: A dictionary containing node pairs as keys and their corresponding shortest path distances as values.
    """
    n = len([x for x in g1.nodes()])
    total_dic = [compute_node_dist(x, g1) for x in g1.nodes()]
    # Flatten the list of dictionaries into a single dictionary.
    total_dic = {k: v for d in total_dic for k, v in d.items()}
    print("First consistency check:", len(total_dic) == n ** 2)
    return total_dic


def compute_line_g_dist_from_g(edge, g1, total_dic):
    """
    Compute the shortest path distances between two edges in a line graph.

    Args:
        edge: The edge for which distances are calculated.
        g1: The input line graph.
        total_dic: A dictionary containing precomputed node distances.

    Returns:
        dict: A dictionary containing edge pairs as keys and their corresponding shortest path distances as values.
    """
    edge_dic_distance = {}
    for edge1 in g1.edges():
        if edge == edge1:
            edge_dic_distance[(edge, edge1)] = 0
        else:
            # Compute the minimum node distance between the two edges.
            node_distance = min([total_dic[x] for x in itertools.product(list(edge), list(edge1))]) + 1
            edge_dic_distance[(edge, edge1)] = node_distance
    return edge_dic_distance


def from_line_graph_to_distance(g1, total_dic):
    """
    Calculate the shortest path distances between all pairs of edges in a line graph.

    Args:
        g1: The input line graph.
        total_dic: A dictionary containing precomputed node distances.

    Returns:
        dict: A dictionary containing edge pairs as keys and their corresponding shortest path distances as values.
    """
    E = len([x for x in g1.edges()])
    line_graph_distance_list = [compute_line_g_dist_from_g(x, g1, total_dic) for x in g1.edges()]
    # Flatten the list of dictionaries into a single dictionary.
    line_graph_distance_voc = {k: v for d in line_graph_distance_list for k, v in d.items()}
    print("Second consistency check:", len(line_graph_distance_voc) == E ** 2)
    return line_graph_distance_voc


### method to focus only on the part of DNC dataset with largest number of contacts

def get_df_connected_from_raw_DNC(large_part = True,t_0 = None,active_thresh = 0):

    """
    Process and filter data from the specified dataset, creating a temporal network

    Args:
        title1 (str): The title of the dataset to process.
        active_thresh (int, optional): The threshold for filtering low-activity links. Defaults to 0.
        remove_outlier (bool, optional): Whether to remove outliers in inter-event times by shifting data. Defaults to True.
        large_part (bool,optional): Wheather to remove the part of DNC dataset with smallest number of nodes
        t_0 (int, opt): reference time to divide the DNC dataset (it is considered only for DNC dataset)
    Returns:
        pd.DataFrame: Processed dataframe containing the filtered and aggregated dataset.
    """
    df = get_df_from_raw('DNC_Mail')
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
