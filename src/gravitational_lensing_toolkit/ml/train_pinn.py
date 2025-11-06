import importlib as _il
_m = _il.import_module('ml.train_pinn')
globals().update({k: v for k, v in _m.__dict__.items() if not k.startswith('_')})
