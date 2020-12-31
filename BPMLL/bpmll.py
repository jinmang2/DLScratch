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

    def forward(self, input, target):
        # https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkde06a.pdf
        # input's shape is (bsz, Q+2)
        # target's shape is (bsz, Q+2)
        # target's unique value is in [-1, 1]
        
        # get true and false labels
        y_i = (target == 1)
        y_i_bar = (target != 1)
        
        # get indices to check
        truth_matrix = torch.logical_and(y_i.unsqueeze(2), y_i_bar.unsqueeze(1))
        # calculate all exponential diff each batches
        exp_matrix = torch.exp(output.unsqueeze(1) - output.unsqueeze(2))
        
        # get normalizing term (length)
        length = truth_matrix.sum(dim=[1,2])
        
        # calculate inner summation term
        inner_sum = self._pad_sequence(exp_matrix[truth_matrix], length).sum(dim=0)
        # apply normalizing terms
        losses = (1 / length) * inner_sum
        
        return losses.sum()
    
    @staticmethod
    def _pad_sequence(sequence, length):
        if isinstance(length, torch.Tensor):
            length = tuple(length.numpy())
        return  pad_sequence(sequence.split(length))
       
