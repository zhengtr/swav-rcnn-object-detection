from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import torch
from torch import Tensor

class CustomizedBoxHead(nn.Module):
    """
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.ln1 = nn.LayerNorm(representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.ln2 = nn.LayerNorm(representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = self.fc6(x)
        x = F.relu(self.ln1(x))
        x = self.fc7(x)
        x = F.relu(self.ln2(x))

        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    https://github.com/pytorch/vision/blob/de31e4b8bf9b4a7e0668d19059a5ac4760dceee1/torchvision/models/_utils.py#L13
    
    Examples::
        >>> m = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed

    Args:
        num_features (int): Number of features ``C`` from an expected input of size ``(N, C, H, W)``
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: Tensor) -> Tensor:
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps})"

class MyBottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, kernel_last=1, expan=4, short_cut=False):
        super(MyBottleneck, self).__init__()
        self.short_cut = short_cut
        self.expansion = expan
        self.conv1 = nn.Conv2d(in_planes, planes, stride=stride, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.norm2 = nn.GroupNorm(32, planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, stride=1, kernel_size=kernel_last, bias=False)
        self.norm3 = nn.GroupNorm(32, self.expansion*planes)

        if short_cut:
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(32, self.expansion*planes)
                )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        if self.short_cut:
            out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResBoxHead(nn.Module):
    def __init__(self):
        super(ResBoxHead, self).__init__()
        self.in_planes = 256

        self.layer1 = MyBottleneck(self.in_planes, 128, 2, expan=4, short_cut=True)
        self.layer2 = MyBottleneck(512, 128, 2, expan=4)
        self.layer3 = MyBottleneck(512, 128, 1, expan=8, kernel_last=2)
        self.norm = nn.GroupNorm(32, 1024)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.norm(out)
        out = out.view(out.size(0), -1)
        return out