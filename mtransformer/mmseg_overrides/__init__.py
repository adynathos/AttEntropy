

def obj_copy_attrs(dest, src):
	for k, v in src.__dict__.items():
		setattr(dest, k, v)