# Re-export ML modules from top-level package
import importlib as _il
for _mod in [
	'ml',
	'ml.pinn',
	'ml.train_pinn',
	'ml.generate_dataset',
	'ml.transfer_learning',
	'ml.evaluate',
]:
	try:
		_m = _il.import_module(_mod)
		globals().update({k: v for k, v in _m.__dict__.items() if not k.startswith('_')})
	except Exception:
		pass


