#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:45:02 2019

@author: jonathanlevine
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../embo')
sys.path.append('../predinfo')
from embo.embo import empirical_bottleneck as eb


def trial_label(x):
    as_strings = [num.astype(str) for num in x]
    return "".join(as_strings)

def make_features(trials_data):
    labeled_data = np.apply_along_axis(trial_label, 0, trials_data)
    combos = np.unique(labeled_data)
    string_to_index = dict(zip(combos, np.arange(len(combos))))
    map_to_index = np.vectorize(lambda x: string_to_index[x])
    mapped_data = map_to_index(labeled_data)
    return mapped_data

def get_marginal(x):
    """
    Helper function to compute and return marginal probability distribution for a 1d vector (x)
    """
    px = np.array([np.sum(x==xi) 
                      for xi in np.unique(x)])/len(x)
    return px

def get_joint(x, y):
    """
    Computes joint probability distribution between 1d vectors x and y
    """
    #  set up dictionary for joint distribution (x-->y-->freq)
    joint_x_y = {}
    
    for x_un in np.unique(x):
        joint_x_y[x_un] = dict(zip(np.unique(y), np.zeros(len(np.unique(y)))))
        
#    populate dictionary 
    for trial, x_val in enumerate(x):
        y_val = y[trial]
        joint_x_y[x_val][y_val] += 1
        
#   normalize to make distirbution  
    joint_sum = sum(sum(list(c.values())) for c in list(joint_x_y.values()))
    
    for key1 in joint_x_y:
        for key2 in joint_x_y[key1]:
            joint_x_y[key1][key2] /= joint_sum
            
    return joint_x_y

def mutual_inf(x, y):
    """
    Calculates the mutual information I(x;y)
    Assuming x,y are both [n x 1] dimensional
    """  
#     Calculate marginal distributions
    px = get_marginal(x)
    py = get_marginal(y)
    
    
    joint_x_y = get_joint(x,y)
# calculate mutual information
    mi = 0
    
    for n_x, x_un in enumerate(np.unique(x)):
        pxi = px[n_x] # p(x)
        
        for n_y, y_un in enumerate(np.unique(y)):
            pyi = py[n_y] # p(y)            
            
            joint_i = joint_x_y[x_un][y_un] # P(x,y)
            
            if ((pxi == 0) or (pyi == 0) or (joint_i ==0 )):
                continue
            else:
                mi += joint_i * np.log2(joint_i/(pxi*pyi))
                
    return mi

def deltaBound(ib_ipast,ib_ifuture,p_ipast,p_ifuture):
    ''' Function to calculate distance from the bound between an empirical IB and participant predictive info
    ib_ipast: ipast of empirical IB (x of convex hull)
    ib_ifuture: ifuture of empirical IB (y of convex hull)
    p_ipast: participant ipast (uncorrected)
    p_ifuture: participant ifuture (uncorrected)
    
    Returns participant ifuture minus the empirical bound (more negative = farther away from the bound)
    '''
    ind = np.argwhere(np.array(ib_ipast) > p_ipast)[0][0]
    slp = (ib_ifuture[ind]-ib_ifuture[ind-1])/(ib_ipast[ind]-ib_ipast[ind-1])
    intercept = ib_ifuture[ind]-(slp*ib_ipast[ind])
    #Return distance between participant Ifuture and interpolated bound - higher 
    return p_ifuture - ((p_ipast*slp)+intercept)


# FUNCTIONS TO COMPUTE CONVEX HULL
def split(u, v, points):
    # return points on left side of UV
    return [p for p in points if np.cross(p - u, v - u) < 0]

def extend(u, v, points):
    if not points:
        return []

    # find furthest point W, and split search to WV, UW
    w = min(points, key=lambda p: np.cross(p - u, v - u))
    p1, p2 = split(w, v, points), split(u, w, points)
    return extend(w, v, p1) + [w] + extend(u, w, p2)

def convex_hull(points):
    # find two hull points, U, V, and split to left and right search
    u = min(points, key=lambda p: p[0])
    v = max(points, key=lambda p: p[0])
    left, right = split(u, v, points), split(v, u, points)

    # find convex hull on each side
    return [v] + extend(u, v, left) + [u] + extend(v, u, right) + [v]

def mi_per_subject(subject_data):
#     trial_data = np.array([subject_data['Reward'].values, subject_data['SubChoice'].values])
    trial_data = np.array([[(subject_data['Reward'].values)[0]], 
                           [(subject_data['SubChoice'].values)[0]], 
                           [(subject_data['RichFrac'].values)[0]]])

    trial_data = trial_data.squeeze()
    features = make_features(trial_data)
    resp = trial_data[1,:]
    
#     1 Back version
    predInfo = [mutual_inf(features[:-1],resp[1:]),mutual_inf(features[1:],resp[:-1])]
    
#   2 Back version
#     predInfo = [mutual_inf(features[:-2],resp[2:]),mutual_inf(features[1:],resp[:-1])]
    
    Fpast = features[:-1]
    Ffuture = features[1:]
    #  >>>>THIS LINE TAKES A LONG TIME TO RUN   
    i_p,i_f,beta,mi,hx,hy = eb(Fpast,Ffuture,numbeta=3000,maxbeta=5, parallel=16)
    #  <<<<< THIS LINE TAKES A LONG TIME TO RUN 
    return (predInfo,i_p,i_f,mi)


def main():
    filename = "Data/RL_MDD_All.csv"
    data_file = pd.read_csv(filename)
    
    depressed = data_file.groupby(['Subject']).agg({'Group': lambda x: x.iloc[0]})
    
    my_concat = lambda x: list(x.values)
    agg_df = data_file.groupby(['Subject']).agg({
        'SubChoice':my_concat,
        'Reward' : my_concat,
        'RichFrac' : my_concat
    }).reset_index()
    
    sub_dict = {}
    N = len(np.unique(agg_df['Subject']))
    for i,subject in enumerate(np.unique(agg_df['Subject'])):
        print('Subject # %s/%s' % (i+1, N))
        sub_dict[subject] = {}
        sub_dict[subject]['Subject'] = subject
        mi,i_p, i_f,saturation = mi_per_subject(agg_df[agg_df['Subject']==subject])
        
        sub_dict[subject]['Ipast'] = mi[0]
        sub_dict[subject]['Ifuture'] = mi[1]
        sub_dict[subject]['i_p'] = i_p
        sub_dict[subject]['i_f'] = i_f
        sub_dict[subject]['saturation'] = saturation
        
    np.save('bounds_one_back.npy', sub_dict)
    
if __name__ == "__main__":
    main()