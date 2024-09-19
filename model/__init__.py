from .SEAVS import AVS_Model

def build_model(type, **kwargs):
    if type =='AVS_Model':
        return AVS_Model(**kwargs)
    else:
        raise ValueError
