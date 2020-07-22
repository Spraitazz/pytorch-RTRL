import torch
import time

from load_data_and_model import load

model, loss_func, optimizer, X, y = load()
n_epochs = 10
for i_ep in range(n_epochs):   
    st = time.time()     
    optimizer.zero_grad() 
    b_s = X.shape[0]   
    T = X.shape[1]
    h = torch.randn(b_s, model.hidden_size, dtype=model.dtype)    
    for t in range(T):
        h = model.h_step(X[:,t].view(-1,1), h)
    y_pred_logits = model.h_to_logits(h)
    loss = loss_func(y_pred_logits, y).mean()     
    loss.backward()
    optimizer.step()
    et = time.time()
    print('epoch {}, loss: {:.6e}, time taken: {:.1f} s'.format(i_ep, loss.item(), et-st))

