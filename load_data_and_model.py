import torch
import numpy as np

from model import RecurrentModel
from make_data import make_data_timeseries_classifier

np.random.seed(69)
torch.manual_seed(69)

def load():
    b_s = 5
    len_s = 1000
    max_T = 50
    dtype = torch.float32
    hidden_size = 10
    
    X, y = make_data_timeseries_classifier(b_s, len_s, max_T, dtype=dtype)
    n_classes = max_T-1    
    model = RecurrentModel(hidden_size, n_classes, dtype)
    #model = model.double() #CHECK IF PRECISION PLAYS A ROLE
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, amsgrad=True)
    
    return model, loss_func, optimizer, X, y 
