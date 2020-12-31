import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class Model(nn.Module):
    
    def __init__(self, num_features, hid_dim, Q):
        super().__init__()
        self.layer1 = nn.Linear(num_features, hid_dim, )
        self.layer2 = nn.Linear(hid_dim, Q+2)
        
    def forward(self, x):
        out = torch.tanh(self.layer1(x))
        return torch.tanh(self.layer2(out))
        
        
class BPMLLLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target, batch_first=True):
        # https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkde06a.pdf
        # input's shape is (bsz, Q+2)
        # target's shape is (bsz, Q+2)
        # target's unique value is in [-1, 1]
        
        # get a batch size
        ix = 0 if batch_first else -1
        bsz = target.size(ix)
        
        # get true and false labels
        y_i = (target == 1)
        y_i_bar = (target != 1)
        
        # get indices to check
        truth_matrix = torch.logical_and(y_i.unsqueeze(2), y_i_bar.unsqueeze(1))
        
        # calculate all exponential diff each batches
        exp_matrix = torch.exp(output.unsqueeze(1) - output.unsqueeze(2))
        
        # get normalizing term
        length = truth_matrix.sum(dim=[1,2])
        
        # padding and sum per batches
        batch_losses = pad_sequence(
            # For batch calculate, do pad by 0 values
            exp_matrix[truth_matrix].split(tuple(length.numpy()))
        ).sum(dim=0)
        
        # normalizing
        losses = batch_losses / y_norm
        
        # calculate loss
        loss = (losses / bsz).sum()
        
        return loss
