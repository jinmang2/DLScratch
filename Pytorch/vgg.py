import torch
import torch.nn
from typing import List, Optional, Union
from copy import copy


# VGG definition that conveniently let's you grab the outputs from any layer
  
class VGG(nn.Module):
    
    def __init__(
        self, 
        in_channels:int = 3, 
        n_layers_per_block:List[int] = [2, 2, 4, 4, 4],
        n_channels_per_block:List[int] = [64, 128, 256, 512, 512]
    ):
        super().__init__()
        loop = zip(n_layers_per_block, n_channels_per_block)
        for ix, (block, channels) in enumerate(loop, start=1):
            out_channels = channels
            for block_ix in range(1, block+1):
                if block_ix == 2: in_channels = channels
                setattr(self, f'conv{ix}_{block_ix}',
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            setattr(self, f'pool{ix}', nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.valid_name = [name for name, _ in self.named_children()]
                   
    def forward(self, x: torch.Tensor, inp: Union[str, List]=[]):
        _inp = self._check_valid_inp(inp)
        out = {}
        for name, layer in self.named_children():
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = torch.relu(x)
            if name in _inp:
                out[name] = x
                _inp.remove(name)
            if len(_inp) == 0:
                break
        return [out[key] for key in inp]
               
    def _check_valid_inp(self, inp):
        if isinstance(inp, str):
            self._is_in_layers(inp)
            inp = [inp]
        elif isinstance(inp, list):
            for i in inp:
                self._is_in_layers(i)
        else:
            raise AttributeError(f'type(inp): str or list')
        if len(inp) == 0:
            inp.append(self.valid_name[-1]) # final layer
        return copy(inp) # from copy import copy
    
    def _is_in_layers(self, name):
        if name not in self.valid_name:
            raise AttributeError(f'{name} not in self.valid_name.')
