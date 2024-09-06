from .RefFormerHead import RefFormerHead, RefFormerHead_res50

def build_head(type, **kwargs):
    if type == 'RefFormerHead':
        return RefFormerHead(**kwargs)
    elif type == 'RefFormerHead_res50':
        return RefFormerHead_res50(**kwargs)
    else:
        raise ValueError


__all__ = ['build_head']
