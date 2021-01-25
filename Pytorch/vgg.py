import torch
import torch.nn
from torch.utils.hooks import RemovableHandle
from typing import List, Optional, Union, Tuple, Callable, Dict
from copy import copy


# VGG definition that conveniently let's you grab the outputs from any layer
  
class VGG(nn.Module):
    
    def __init__(
        self, 
        in_channels: int=3, 
        n_layers_per_block: List[int]=[2, 2, 4, 4, 4],
        n_channels_per_block: List[int]=[64, 128, 256, 512, 512],
        pool: str='max',
    ):
        super().__init__()
        
        assert len(n_layers_per_block) == len(n_channels_per_block)
        self.num_blocks = len(n_layers_per_block)
        
        loop = zip(n_layers_per_block, n_channels_per_block)
        for ix, (block, channels) in enumerate(loop, start=1):
            out_channels = channels
            for block_ix in range(1, block+1):
                if block_ix == 2: in_channels = channels
                setattr(self, f'conv{ix}_{block_ix}',
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            pool = 'MaxPool2d' if pool.lower() == 'max' else 'AvgPool2d'
            setattr(self, f'pool{ix}', getattr(nn, pool)(kernel_size=2, stride=2))
            in_channels = out_channels        
        
    def forward(self, x: torch.Tensor):
        for layer in self.children():
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = torch.relu(x)
        return x

            
class FeatureExtractor(nn.Module):
    
    # https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.valid_name = [name for name, _ in self.model.named_children()]

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        self.handle[layer_id].remove()
        return fn
      
    def forward(self, x: torch.Tensor, layers: List[str]) -> Dict[str, torch.Tensor]:
        self._check_valid_inp(layers)
        self.handle = {layer_id: RemovableHandle(lambda x: _) for layer_id in layers}
        self._features = {layer_id: torch.empty(0) for layer_id in layers}
        for layer_id in layers:
            layer = dict(self.model.named_children())[layer_id]
            hook = self.save_outputs_hook(layer_id)
            self.handle[layer_id] = layer.register_forward_hook(hook)
        _ = self.model(x)
        return [self._features[layer_id] for layer_id in layers]
    
    def _check_valid_inp(self, inp):
        if isinstance(inp, str):
            self._is_in_layers(inp)
            inp = [inp]
        elif isinstance(inp, list):
            for i in inp:
                self._is_in_layers(i)
        else:
            raise AttributeError(f'type(inp): str or list')
        return copy(inp) # from copy import copy
    
    def _is_in_layers(self, name):
        if name not in self.valid_name:
            raise AttributeError(f'{name} not in self.valid_name.')
