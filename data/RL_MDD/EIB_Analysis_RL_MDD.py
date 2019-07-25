import sys, csv
import numpy as np
import pandas as pd
sys.path.append('../../embo')
sys.path.append('../../predinfo')
from embo.embo import empirical_bottleneck as eb
from predinfo import predinfo
from numba import jit
from scipy import stats
from scipy.stats import spearmanr, entropy
from matplotlib.pyplot import *
import pickle

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

#Helper function for calculating eib
def getEIB(features,numbeta=10000,maxbeta=50):
    '''
    Function to calculate empirical bound for a subset of features
    for now, the size of the feature lambda used for the bound calculation is equal to the number of unique features contained within the feature sequence
    
    Input:
    features: array of features obserbved in the experiment
    numbeta: number of beta subdivision for the IB calculation between 0 and maxbeta
    maxbeta: maximum value of beta to be considered
    '''
    
    # For now, size of feature lambda is equal to the number of features present in the sequence
    f_un = np.unique(features)
    for i in range(len(f_un)):
        features[features == f_un[i]] = i
    
    # Calculate empirical bound
    i_p,i_f,beta,mi,hx,hy = eb(features[0:len(features)-1],features[1:len(features)],numbeta=numbeta,maxbeta=maxbeta,parallel = 8)
    return i_p,i_f,beta,mi,hx,hy

# Function to plot EIB for a specific sequence and add participant predictive info
def plotEIB(eib,ips,ifs,mi,hx,t,col,trim=1, ylab=True,xlab=True):
    plot([0,hx],[mi,mi],'--'+col,alpha=.6,lw=1.5)
    plot([hx,hx],[0,mi],'--'+col,alpha=.6,lw=1.5)
    plot(eib[0,:],eib[1,:],'-'+col,lw=1.5)
    plot(ips,ifs,'o'+col,markersize=7)
    title(t)
    if ylab == True:
        ylabel('$I_{future}$')
    if xlab == True:
        xlabel('$I_{past}$')

#Import the subject data from the experiment
data = pd.read_csv('../../data/RL_MDD/RL_MDD_All.csv',sep=',')

#Fix for 1,2 convention
data['RichFrac'] = data['RichFrac'] - 1
#Fix for Frac choice convention
data['RichFracChoice'] = data['RichFracChoice'] ^ 1

#Define Feature Space
data['Feature'] = np.zeros((data.shape[0],1))
data['Feature'] = data['RichFrac'] + (data['RichFracChoice'] * 2) + (data['Reward'] * 4)

#Get a list of all the subjects
subjIDs = data['Subject'].unique()

#Compute the Information Bottleneck curve - whole experiment i.e. Reward and Punishment runs concatenated
nb = 1000 #number of betas
mb = 100 #max value of beta
data_eibs = {}
hulls = {}
###########This is the loop it doesn't get through
for subject in subjIDs:
    print("calculating EIB for " + str(subject))
    features = data['Feature'][data['Subject']==subject]
    data_eibs[subject] = getEIB(features,numbeta=nb,maxbeta=mb)
    print("done")

    #Get convex hulls for each empirical bottleneck 
    eib = data_eibs[subject]
    hulls[subject] = np.vstack((eib[0],eib[1]))


with open('eibs_RLLMDD.pkl','wb') as handle:
    pickle.dump(hulls, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Compute Ipast and Ifuture for all the subjects
datainfo = pd.DataFrame({'Subject' : subjIDs})
#dim = datainfo['Subject'].size
datainfo['Ipast'] = np.zeros_like(datainfo['Subject'])
datainfo['Ifuture'] = np.zeros_like(datainfo['Subject'])
'''
for subject in datainfo['Subject']:
    #Get data for specific subject
    subjdata = data[data['Subject']==subject]

    #Get vector of features observed by subject
    feat = subjdata['Feature']

    #Get vector of subject responses
    resp = subjdata['RichFracChoice']

    #Calculate Ipast and Ifuture for this subject
    Ipast, Ifuture = predinfo(resp[1:], feat[:-1], 2,1,2,8)

    datainfo['Ipast'][datainfo['Subject'] == subject] = Ipast
    datainfo['Ifuture'][datainfo['Subject'] == subject] = Ifuture
'''


#datainfo.to_pickle('./datainfo_RLMDD.pkl')
#datainfo = pd.read_pickle('./datainfo_RLMDD.pkl')
'''
for subject in subjIDs:
    figure()
    hull = hulls[subject]
    plot(hull[0,:],hull[1,:])
    ipast = datainfo['Ipast'][data['Subject']==subject]
    ifuture = datainfo['Ifuture'][data['Subject']==subject]
    plot(ipast,ifuture,marker='o',markersize=7,linestyle='None')
    title('All')
    ylabel('$I_{future}$')
    xlabel('$I_{past}$')

    savefig('/home/gjf/RL_MDD_fig_' + str(subject) + '.eps')
'''
