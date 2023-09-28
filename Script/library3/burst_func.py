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

def burst_train(inter_ev, dt):
    """
    Calculate burst train statistics from a list of inter-event times.

    Args:
        inter_ev (list): List of inter-event times.
        dt (float): Time interval for burst detection.

    Returns:
        tuple: A tuple containing burst counts and burst distribution.
    """
    ev_distr = np.zeros(len(inter_ev))
    i = 0

    cnt = Counter()
    d = 0
    c = 1
    ev_distr[0] = d
    
    while i < len(inter_ev):
        if inter_ev[i] <= dt:
            ev_distr[i + 1] = d
            c += 1
            i += 1
        else:
            cnt.update(Counter([c]))
            d += 1
            if i < len(inter_ev) - 1:
                ev_distr[i + 1] = d
            i += 1
            c = 1
            continue
    
    # Assert that the burst counts match the burst distribution sum
    assert (sum([k * v for k, v in cnt.items()]) == np.sum([k * v for k, v in Counter(Counter(ev_distr).values()).items()]))
    
    return cnt, ev_distr

def get_burst_train(df, dt):
    """
    Calculate burst train statistics for a DataFrame of events.

    Args:
        df (DataFrame): DataFrame containing event data.
        dt (float): Time interval for burst detection.

    Returns:
        tuple: A tuple containing burst counts and the modified DataFrame with burst information.
    """
    df_rest = df.copy()
    int_ev_time = [x for x in (df_rest.timestamp.shift(-1) - df_rest.timestamp).values if ((x != np.nan))]
    b_train = burst_train(int_ev_time, dt)[0]
    bursts = burst_train(int_ev_time, dt)[1]
    df['burst'] = np.array(bursts)
    
    return Counter(b_train), df



def compute_bursty_trains(df, line_graph, dt_list, graph_list, seed, title1, include_single_link_burst=False, reproduce_paper=False):
    """
    Calculate bursty trains for a given DataFrame and save the results.

    Args:
        df (DataFrame): Input DataFrame containing temporal network data.
        line_graph (Graph): Graph representing the network.
        dt_list (list): List of time intervals to consider.
        graph_list (list): List of graphs to use in computations (e.g., ['G'], ['G1', 'G2', 'G3']).
        seed (int): Random seed for shuffling data.
        title1 (str): Title for the results.
        include_single_link_burst (bool): Include burst counts for single links.
        reproduce_paper (bool): Reproduce the paper's results.

    Returns:
        dict: A dictionary containing the computed bursty trains.
    """
    # Check assertions based on the input parameters
    assert graph_list
    if reproduce_paper:
        assert (graph_list == ['G']) or (graph_list == ['G1', 'G2', 'G3'])
        assert include_single_link_burst

    # Initialize dictionaries to store the results
    overall_dictionary = dict()
    df_dictionary = dict()

    # Loop through the graph list and initialize sub-dictionaries
    for k in graph_list:
        overall_dictionary[k] = dict()
        if include_single_link_burst:
            overall_dictionary[k]['burst_cnt_link'] = dict()
        overall_dictionary[k]['burst_cnt_ego'] = dict()
        overall_dictionary[k]['n_link_cnt'] = dict()

        # Create a copy of the DataFrame for each graph type
        if k == 'G':
            df_dictionary[k] = df.copy()
        elif k == 'G1':
            df_dictionary[k] = permute_timestamps(df, seed)  # Timestamp reshuffle
        elif k == 'G2':
            df_dictionary[k] = shuffle_df(df, seed)  # Time series reshuffle
        elif k == 'G3':
            df_dictionary[k] = random_df_same_weight(df, seed)  # Time series reshuffle with same n contacts

    # Loop through the specified time intervals
    for dt in dt_list:
        dt_dictionary = dict()

        # Loop through the graph list for each time interval
        for k in graph_list:
            dt_dictionary[k] = dict()
            if include_single_link_burst:
                dt_dictionary[k]['burst_cnt_link'] = Counter()
            dt_dictionary[k]['burst_cnt_ego'] = Counter()
            dt_dictionary[k]['n_link_cnt'] = Counter()

        # Iterate through unique nodes in the network
        for nodes in tqdm(df.nodes.unique()):
            neigh_set = (set(list(nx.neighbors(line_graph, nodes))) | set([nodes]))

            # Loop through the graph list for each node
            for k in graph_list:
                df_graph = df_dictionary[k]

                if include_single_link_burst:
                    df_link = df_graph[df_graph.nodes == nodes]
                    brst_cnt_link, _ = get_burst_train(df_link, dt)
                    dt_dictionary[k]['burst_cnt_link'].update(brst_cnt_link)

                df_ego = df_graph[df_graph.nodes.isin(neigh_set)]
                brst_cnt_ego, df_ego = get_burst_train(df_ego, dt)
                dt_dictionary[k]['burst_cnt_ego'].update(brst_cnt_ego)

                cnt_n_links_per_train = Counter(
                    df_ego.groupby('burst')['nodes'].apply(lambda x: (float(len(set(x))), float(len(list(x))))).values)
                dt_dictionary[k]['n_link_cnt'].update(cnt_n_links_per_train)

        # Store the results in the overall dictionary
        for k in graph_list:
            if include_single_link_burst:
                overall_dictionary[k]['burst_cnt_link'][dt] = dt_dictionary[k]['burst_cnt_link']
            overall_dictionary[k]['burst_cnt_ego'][dt] = dt_dictionary[k]['burst_cnt_ego']
            overall_dictionary[k]['n_link_cnt'][dt] = dt_dictionary[k]['n_link_cnt']

        # If reproducing the paper, save the results to files
        if reproduce_paper:
            if graph_list == ['G']:
                file_2_save = [overall_dictionary[k]['burst_cnt_link'][dt], overall_dictionary[k]['burst_cnt_ego'][dt],
                               Counter(), Counter(), Counter(), Counter()]
                file_2_save_link = [overall_dictionary[k]['n_link_cnt'][dt], Counter(), Counter(), Counter()]

                try:
                    os.mkdir('../Results/Bursty_trains/' + title1 + '/')
                except:
                    print('../Results/Bursty_trains/' + title1 + '/')

                title_burst_cnt = '../Results/Bursty_trains/' + title1 + '/' + title1 + '_' + str(dt)
                assert file_2_save[:2] == joblib.load(title_burst_cnt + '.joblib')[:2]

                title_link_cnt = '../Results/Bursty_trains/' + title1 + '/' + title1 + '_n_links_' + str(dt)
                assert file_2_save_link[0] == joblib.load(title_link_cnt + '.joblib')[0]

            elif graph_list == ['G1', 'G2', 'G3']:
                # The order of the list is weird, but this is the same as the code used for computing results in the paper.
                # It's present only in the option to reproduce the paper.
                file_2_save = [overall_dictionary['G2']['burst_cnt_ego'][dt], overall_dictionary['G1']['burst_cnt_link'][dt],
                               overall_dictionary['G1']['burst_cnt_ego'][dt], overall_dictionary['G3']['burst_cnt_ego'][dt]]
                file_2_save_link = [overall_dictionary['G2']['n_link_cnt'][dt], overall_dictionary['G1']['n_link_cnt'][dt],
                                    overall_dictionary['G3']['n_link_cnt'][dt]]

                try:
                    os.mkdir('../Results/Bursty_trains/' + title1 + 'rand/')
                except:
                    print('../Results/Bursty_trains/' + title1 + 'rand/')

                title_burst_cnt = '../Results/Bursty_trains/' + title1 + 'rand/' + title1 + '_' + str(dt) + '_' + str(seed)
                assert file_2_save == joblib.load(title_burst_cnt + '.joblib')

                title_link_cnt = '../Results/Bursty_trains/' + title1 + 'rand/' + title1 + '_n_links_' + str(dt) + '_' + str(seed)
                assert file_2_save_link == joblib.load(title_link_cnt + '.joblib')

    return overall_dictionary




def event_rate(period,df):
    df1 = df.copy()
    df1.timestamp = df1.timestamp%(period+1)
    strenght = float(df1.shape[0])
    period = float(period)
    return (period/strenght)*df1.groupby(df1.timestamp).size()