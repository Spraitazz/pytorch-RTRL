import numpy as np
import torch

#a toy series - a sequence of zeros with ones repeating with a given period
#output: period of repetition of 1
#repetition period can be between 2 and max_T (inclusive)
def make_data_timeseries_classifier(b_s, len_s, max_T, dtype=torch.float32):     
    Ts = np.random.randint(2,max_T+1, size=b_s)
    n_classes = max_T-1
    X = torch.zeros((b_s, len_s), dtype=dtype)
    for j in range(b_s):
        X[j, [i for i in range(0,len_s,Ts[j])]] = 1.    
    y = torch.tensor(Ts, dtype=torch.int64)-2
    return X, y
