from .TemporalFuserBridger import TemporalFusers, TemporalFusers_res50

def build_bridger(type, **kwargs):
    if type == 'TemporalFusers':
        return TemporalFusers(**kwargs)
    elif type == 'TemporalFusers_res50':
        return TemporalFusers_res50(**kwargs)
    else:
        raise ValueError