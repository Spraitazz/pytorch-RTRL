import torch
import torch.nn.functional as F

class RecurrentModel(torch.nn.Module):
    def __init__(self, hidden_size, n_classes, dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        
        self.lin_encoder = torch.nn.Linear(1, 100)
        self.rnn = torch.nn.GRUCell(100, hidden_size) #, nonlinearity='relu')
        #self.rnn.state_dict()['weight_hh'] *= 0.001
        self.lin_logits = torch.nn.Linear(hidden_size, n_classes)
        
    def x_encode(self, x):
        code = self.lin_encoder(x)        
        return code
        
    def h_to_logits(self, h):
        logits = self.lin_logits(h)
        return logits
        
    def h_step(self, x, h):
        code = self.x_encode(x)
        h_next = self.rnn(code, h)
        return h_next
