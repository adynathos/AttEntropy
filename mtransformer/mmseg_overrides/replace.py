from mmseg.models.utils.embed import PatchEmbed
from . import obj_copy_attrs

def replace_module(mod, path, dest, verbose=True):
	path_segments = path.split('.')
	child_name = path_segments[-1]
	parent_path = '.'.join(path_segments[:-1])
	if verbose:
		print(f'	replace {parent_path} child {child_name}')
	parent_mod = mod.get_submodule(parent_path)
	dest.module_path = path
	# parent_mod.add_module(child_name, dest)
	setattr(parent_mod, child_name, dest)
	
def replace_modules(mod, src_class, fn, verbose=False):
	"""
	@param fn: function transorming old module into new.
	"""
	if verbose:
		print(f'Replacing mods of class {src_class.__name__}:')

	for path, submod in mod.named_modules():
		if isinstance(submod, src_class):
			replace_module(mod, path, fn(submod), verbose=verbose)
		
	if verbose:
		print()

class PatchEmbed_ViewShapes(PatchEmbed):
	def forward(self, x):	
		res = super().forward(x)
		print(self.module_path, f'PatchEmbed insh={tuple(x.shape)} out={tuple(res[0].shape)}, grid shape {res[1]}')
		return res
		
	@classmethod
	def from_superclass(cls, module_orig):
		module_new = cls()
		obj_copy_attrs(module_new, module_orig)	
		return module_new





	
