import time

from load_data_and_model import load
from train_epoch_rtlr import *

model, loss_func, optimizer, X, y = load()
n_epochs = 10
for i_ep in range(n_epochs):
    st = time.time()
    loss = train_epoch(model, loss_func, optimizer, X, y) 
    et = time.time()   
    print('epoch {}, loss: {:.6e}, time taken: {:.1f} s'.format(i_ep, loss, et-st))
