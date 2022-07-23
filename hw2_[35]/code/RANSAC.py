import numpy as np
import random

def RANSAC(feature_pairs):
    n = 1
    p = 0.7
    P = 0.99
    
    #k=class teach or len(pair)
    #k_times = int(math.log(1-P)/math.log(1-p**n))
    k_times = len(feature_pairs)
    max_inlier = 0
    chosen_shift = []
    feature_pairs = np.asarray(feature_pairs)
    for k_ in range(k_times):
        #draw n samples randomly
        #random_idx = random.randint(0,feature_pairs.shape[0]-1)
        #random_sample = feature_pairs[random_idx]
        
        random_sample = feature_pairs[k_]
        
        #fit para with n samples
        shift = random_sample[1] - random_sample[0]
        
        #calculate number of other sample which fit model
        inlier_cnt = 0
        for feature_pair in feature_pairs:
            diff = shift-(feature_pair[1] - feature_pair[0])
            
            if np.sqrt((diff**2).sum()) < 10:
                inlier_cnt += 1
                
        #choose largest fit
        if inlier_cnt > max_inlier and shift[0]<0:
            max_inlier = inlier_cnt
            chosen_shift = shift

    #calculate relative shift
    chosen_shift = list(chosen_shift)
    chosen_shift[1] *= -1
    
    return chosen_shift
