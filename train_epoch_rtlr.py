import torch
import time
import copy

from vector_to_params_grad import add_vector_to_parameters_grad

def pass_example(model, loss_func, x, y): 

    model_clone = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model_clone.parameters(), 0.) #used for zero.grad() function only here
    T = x.shape[0]    
    theta = torch.nn.utils.convert_parameters.parameters_to_vector(model_clone.parameters())      
    n_params = len(theta)
    
    h = torch.randn(model.hidden_size, dtype=model.dtype, requires_grad=True)
    dh_dtheta = torch.zeros((model.hidden_size, n_params), dtype=model.dtype) 
    dhnext_dhprev = torch.zeros((model.hidden_size, model.hidden_size), dtype=model.dtype)
    partial_dh_dtheta = torch.zeros_like(dh_dtheta)
    
    for t in range(T): 
        h_next = model_clone.h_step(x[t].view(1,1), h.view(1, model.hidden_size)).view(model.hidden_size)
        #compute dh/dhprev and partial dh/dparams
        for i_h in range(model.hidden_size):
            v = torch.zeros(model.hidden_size, dtype=model.dtype)
            v[i_h] = 1.  
            if i_h == model.hidden_size-1:
                h_next.backward(v) 
            else:
                h_next.backward(v, retain_graph=True) 
            dhnext_dhprev[i_h] = h.grad.clone()  
            h.grad = None             
            grad_generator = (param.grad if param.grad is not None else torch.zeros_like(param) for param in model_clone.parameters())                         
            theta_grad = torch.nn.utils.convert_parameters.parameters_to_vector(grad_generator)                         
            partial_dh_dtheta[i_h] = theta_grad.clone()                
            optimizer.zero_grad()        
        dh_dtheta = torch.mm(dhnext_dhprev, dh_dtheta) + partial_dh_dtheta                  
        h_next = h_next.detach()
        h = h_next.clone()            
        h.requires_grad = True           
   
    y_pred = model_clone.h_to_logits(h.view(1, model.hidden_size))
    loss = loss_func(y_pred, y.view(1))    
    loss.backward()        
    #add partial derivative of loss wrt. params and (loss wrt h) times (h wrt params)
    grad_generator = (param.grad if param.grad is not None else torch.zeros_like(param) for param in model_clone.parameters())            
    partial_theta_grad = torch.nn.utils.convert_parameters.parameters_to_vector(grad_generator)    
    theta_grad = partial_theta_grad.clone() + h.grad.clone() @ dh_dtheta   
    return loss.item(), theta_grad
    
    
#https://github.com/petered/uoro-demo/blob/master/uoro_demo/rtrl.py
def train_epoch(model, loss_func, optimizer, X, y):
    
    n_X = X.shape[0]    
    theta_grads = torch.zeros_like(torch.nn.utils.convert_parameters.parameters_to_vector(model.parameters()))    
    avg_loss = 0.        
    
    for i_x in range(n_X):
        t_s = time.time()        
        loss, theta_grad = pass_example(model, loss_func, X[i_x], y[i_x])
        theta_grads += theta_grad/n_X   
        avg_loss += loss            
        t_e = time.time()
        print('passed example {} in {:.1f} s. loss: {}'.format(i_x, t_e-t_s, loss))
       
    #set gradients and take optimization step        
    for param in model.parameters():    
        param.grad = torch.zeros_like(param)
    add_vector_to_parameters_grad(theta_grads, model.parameters())    
    optimizer.step()
    optimizer.zero_grad()
    return avg_loss / n_X
