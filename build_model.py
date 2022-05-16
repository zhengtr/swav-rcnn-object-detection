import customized_module as my_nn
import torch
import torch.nn as nn

def replace_bn_FBN(m, name):    
    """
        Warning: not a generalized verion! 
        Use it only if you know what you are doing.
    """
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)

        if type(target_attr) == torch.nn.BatchNorm2d:
            setattr(m, attr_str, my_nn.FrozenBatchNorm2d(target_attr.num_features, target_attr.eps))
    for n, ch in m.named_children():
        if isinstance(ch, nn.BatchNorm2d):
            m[1] = my_nn.FrozenBatchNorm2d(m[1].num_features, m[1].eps)
        replace_bn_FBN(ch, n)


def replace_bn_GN(m, name):    
    """
        Warning: not a generalized verion! 
        Use it only if you know what you are doing.
    """
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)

        if type(target_attr) == nn.BatchNorm2d:
            setattr(
                m, 
                attr_str, 
                nn.GroupNorm(
                    num_groups=32, 
                    num_channels=target_attr.num_features, 
                    eps=target_attr.eps, 
                    device=torch.device("cuda")
                )
            )
    for n, ch in m.named_children():
        if isinstance(ch, nn.BatchNorm2d):
            m[1] = nn.GroupNorm(
                num_groups=32, 
                num_channels=m[1].num_features, 
                eps=m[1].eps, 
                device=torch.device("cuda")
            )
        replace_bn_GN(ch, n)


def replace_bn_IN(m, name):    
    """
        Warning: not a generalized verion! 
        Use it only if you know what you are doing.
    """
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)

        if type(target_attr) == nn.BatchNorm2d:
            setattr(
                m, 
                attr_str, 
                nn.InstanceNorm2d(num_features=target_attr.num_features),
            )
    for n, ch in m.named_children():
        if isinstance(ch, nn.BatchNorm2d):
            m[1] = nn.InstanceNorm2d(num_features=m[1].num_features)
        replace_bn_IN(ch, n)