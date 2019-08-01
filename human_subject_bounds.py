import sys
sys.path.append('./embo')
sys.path.append('./predinfo')
from embo.embo import empirical_bottleneck as eb
import embo.utils
import numpy as np
from matplotlib import pyplot as plt
import time






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


def get_new_index(seq, bases):
    return np.dot(seq,bases)


"""
feat: feature vector
f: sliding window size for feat
"""


def get_new_features_sliding(feat, f):
    
    n_feat = len(np.unique(feat))
    
    base_feat = n_feat**(np.arange(f))
    
     
    # create new feature vector
    # so that looking 1 into the past or future is equivalent
    # to looking f into the past or future
    # each unique seqeuence of length f gets a new unique integer label 
    # the resulting feature vector will be of length len(feat)-(f+1)
#     if includef0:
#         new_features = np.array([get_new_index(feat[i-f:i+1], base_feat) for i in range(f,len(feat))])
#     else:
        # the resulting feature vector will be of length len(feat)-(f+1)
    new_features = np.array([get_new_index(feat[i-f:i], base_feat) for i in range(f,len(feat))])
    
    return new_features


"""
turns the feature into a string of all of the features in it
so that we can find unique combinations
of the following variables:
    R1 = first choice (1 or 2)
    R2 = second choice (1 or 2)
    RW = reward (0 or 1)
    S2 = second state (1 or 2)
    R1*/best_R1 = first choice that would have led to greatest pr(reward)
"""


def trial_label(x):
    as_strings = [num.astype(str) for num in x]
    return "".join(as_strings)


"""
Data is input in the form:
    [trials x n_features]
    
Data is turned into the form:
    [nTrials X index ]
where index is a unique ID from corresponding to a unique
combination of features
"""

def make_features(trials_data):
    labeled_data = np.apply_along_axis(trial_label, 0, trials_data)
    combos = np.unique(labeled_data)
    string_to_index = dict(zip(combos, np.arange(len(combos))))
    map_to_index = np.vectorize(lambda x: string_to_index[x])
    mapped_data = map_to_index(labeled_data)
    return mapped_data


def main():

    print('READING IN DATA\n')
    read_data = np.load('data/Daw_MBMF/subject_data_for_cluster.npy').item()
    # Still need to work out best_R1 in this data set
    keys = ['best_R1', 'R1', 'S2', 'Rw']
    #keys = ['S2','R1','R2','Rw'] #smaller size to test
    parallel = 16 #AF: change this to an integer to specify the number of cores available - False if you don't want to use parallel processing

    # THIS IS MAIN LOOP OVER different models or weighting params
    print('COMPUTING BOUND')
    start = time.time()
    for w in read_data:
        print('Computing bound %f...'%w)
        subject = read_data[w]

        trials = np.vstack([subject[key] for key in keys])
        features = make_features(trials)


        # Uncomment this for no sliding window
        Fpast = features[:-1]
        Ffuture = features[1:]


        # TODO: Make Sliding Windows Here
        """
        window_size = 1
        Fpast = get_new_features_sliding(features, window_size)
        Ffuture = features[window_size:]
        """

        i_p_emp,i_f_emp,beta,mi,hx,hy = eb(Fpast, Ffuture, numbeta=1000, maxbeta=20,parallel = parallel) #AF: parallel distributes the beta calculations across the number of specified cores




        points = np.stack([i_p_emp,i_f_emp],axis=1)
        hull = np.array(convex_hull(points))
        
        #AF: Function to save the computed points
        np.save('./subject_MBMF_bounds/EIB_W_%f'%w,hull)
    
        #Plot the hull
#         plt.figure()
#         #plt.plot(hull[0:-1,0],hull[0:-1,1],'-o')
#         plt.plot(hull[0:-1,0],hull[0:-1,1],'-')
#         plt.plot(hull[0:-1,0],hull[0:-1,1],'ok')
#         plt.title('empirical bound for w = %.2f' % w)
#         plt.ylabel('$I_{future}$')
#         plt.xlabel('$I_{past}$')
#         plt.savefig('./BoundFigs/MBMF_EIB_W_%f.pdf'%w)
    print(time.time()-start)
    print('\nFINISHED')

if __name__ == '__main__':
    main()


