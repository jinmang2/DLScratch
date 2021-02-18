import torch
import torchvision.models as models



resnets = [
    'ResNet', 'resnet18', 'resnet34', 
    'resnet50', 'resnet101', 'resnet152',
]

class ResNet(nn.Module):

    """
    ResNet 호출할 Class
    Todo:
        - Conv Layer만 떼서 학습
    """
    
    def __init__(
        self,
        n_classes: int, 
        modelname: str='resnet152',
        freeze: bool=True
    ):
        super().__init__()
        if name in resnets:
            self.resnet = getattr(models, modelname)(pretrained=True)
        else:
            raise AttributeError(f"resnet은 다음 중 하나 {resnets}")
        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False
        n_inputs = self.resnet.fc.out_features # 1000
        # 학습시킬 파라미터
        self.fc1 = nn.Linear(n_inputs, 1024)
        self.fc2 = nn.Linear(1024, n_classes)
        
    def forward(
        self, 
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.resnet(x)
        x = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(x))  