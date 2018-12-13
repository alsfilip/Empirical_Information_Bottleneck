from numba import jit
import numpy as np
from math import log

'''
Functions to calculate predictive information and participant distance from the empirical bound

By: Alex Filipowicz (alsfilip@gmail.com)

Code not super optimized but works for now.


Function to calculate predictive information

	This function takes the following variables:

	- resp: 1xN array of participant responses
	- feat: 1xN array of stimulus features
	- resplen: window of symbols to be considered 
	- featlen: window of features to be considered
	- numrespsymbols: number of possible symbols in the responses
	- numfeatsymbols: number of possible symbols in the features

	This function returns:

	- Ipast: the mutual information between past stimulus features and participant responses
	- Ifuture: mutual information between current participant responses and future stimulus features

	Note that Ipast cannot exceed the mutual information between past and present stimulus features, and Ifuture cannot exceed Ipast.

'''
@jit
def predinfo(resp,feat,resplen,featlen,numrespsymbols,numfeatsymbols):

    #Number of unique words that can be made given the participant responses/features window lengths
    numrespwords = numrespsymbols**resplen
    numfeatwords = numfeatsymbols**featlen
    
    #CALCULATE PROBABILITY DISTRIBUTIONS REQUIRED FOR MUTUAL INFORMATION CALCULATION
    
    #Bases used to index the different distributions
    aux_base_resp = numrespsymbols**np.arange(resplen)
    aux_base_feat = numfeatsymbols**np.arange(featlen)
    
    #Initialize response, feature, and joint distributions
    presp = np.zeros(numrespwords) #P(R)
    pfeat = np.zeros(numfeatwords) #P(F)
    prespfeat = np.zeros(numfeatwords*numrespwords) #P(R,F) - for Ifuture
    pfeatresp = np.zeros(numrespwords*numfeatwords) #P(F,R) - for Ipast   
    
    #Count frequences of response words occuring for P(R)
    for i in range(len(resp)-(resplen-1)):
        r = resp[i:i+resplen] #Response 'word' based on the desired response word length
        presp[int(sum(np.flip(r,0)*aux_base_resp))] += 1 #Add a count to the index that corresponds to the response word
    
    #Count frequencies of feature words occuring for P(F)
    feat = feat.astype(int)
    for i in range(len(feat)-(featlen-1)):
        f = feat[i:i+featlen] #Feature 'word' based on the desired feature word length
        pfeat[int(sum(np.flip(f,0)*aux_base_feat))] += 1 #Add a count to the index that corresponds to the feature word
    
    #Count frequencies of response/feature joint distribution for P(R,F) (Ifuture)
    for i in range(len(feat)-(featlen+resplen-1)):
        r = resp[i:i+resplen] #Response 'word' based on the desired response word length
        f = feat[i+resplen:i+resplen+featlen] #Feature 'word' that follows the response word
        #Count the response/feature conjunctions
        prespfeat[int((sum(np.flip(r,0)*aux_base_resp)*numfeatwords)+sum(np.flip(f,0)*aux_base_feat))] += 1
        
    #Count frequencies of feature/response joint distribution for P(F,R) (Ipast)
    for i in range(len(resp)-(resplen+featlen-1)):
        f = feat[i:i+featlen] #Feature 'word' based on the desired feature word length
        r = resp[i+featlen:i+featlen+resplen] #Response 'word' that follows the feature word
        #Count the feature/response conjunctions
        pfeatresp[int((sum(np.flip(f,0)*aux_base_feat)*numrespwords)+sum(np.flip(r,0)*aux_base_resp))] += 1

    #Convert counts to probability distributions
    presp = presp/sum(presp) #P(R)
    pfeat = pfeat/sum(pfeat) #P(F)
    prespfeat = prespfeat/sum(prespfeat) #P(R,F)
    pfeatresp = pfeatresp/sum(pfeatresp) #P(F,R)
    
    #CALCULATE IPAST AND IFUTURE
    Ipast = 0
    Ifuture = 0
    for fi in range(len(pfeat)):
        pf = pfeat[fi] #P(F = f)
        for ri in range(len(presp)):
            pr = presp[ri] #P(R = r)
            
            #To calculate Ipast
            if pf > 0. and pfeatresp[(fi*numrespwords)+ri] > 0. and pr > 0.:
                pfr = pfeatresp[(fi*numrespwords)+ri]/pf #P(f|r) = P(f,r)/p(f)
                Ipast += pf*(pfr*log(pfr/pr,2))
            
            #To calculate Ifuture
            if pf> 0. and prespfeat[(ri*numfeatwords)+fi] > 0. and pf > 0.:
                prf = prespfeat[(ri*numfeatwords)+fi]/pf #P(r|f) = P(r,f)/p(f)
                Ifuture += pf*(prf*log(prf/pr,2))
                
    return Ipast,Ifuture
 