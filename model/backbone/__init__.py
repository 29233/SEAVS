from .resnet import B2_ResNet
from .pvt import pvt_v2_b5
from .pvt_lora import pvt_v2_lora

def build_backbone(type, **kwargs):
    if type == 'res50':
        return B2_ResNet(**kwargs)
    elif type=='pvt_v2_b5':
        return pvt_v2_b5(**kwargs)
    elif type == 'pvt_v2_lora':
        return pvt_v2_lora(**kwargs)
    else:
        raise ValueError
    
    
__all__=['build_backbone']
