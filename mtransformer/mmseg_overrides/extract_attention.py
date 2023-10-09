from functools import partial
from mmcv.cnn.bricks.transformer import MultiheadAttention
from .replace import replace_modules, obj_copy_attrs, PatchEmbed, PatchEmbed_ViewShapes
import torch

from mmpretrain.models.utils.attention import MultiheadAttention as MultiheadAttention_mmpretrain

from mmcv.cnn.bricks.transformer import MultiheadAttention as MultiheadAttention_mmcv

"""

Inspecting a MMPretrain's MAE network

	MAEViT(
	(patch_embed): PatchEmbed(
		(adaptive_padding): AdaptivePadding()
		(projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
	)
	(drop_after_pos): Dropout(p=0, inplace=False)
	(layers): ModuleList(
		(0-11): 12 x TransformerEncoderLayer(
		(ln1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
		(attn): MultiheadAttention(
			....
		)

We check the class of `net.backbone.layers[0].attn` is `mmpretrain.models.utils.attention.MultiheadAttention`,
this is different from mmcv.cnn.bricks.transformer.MultiheadAttention found earlier in MMSegmentation.




"""

"""


## Constructors:


* mmcv has this constructor

mmcv.cnn.bricks.transformer.MultiheadAttention(
    embed_dims,
    num_heads,
    attn_drop=0.0,
    proj_drop=0.0,
    dropout_layer={'type': 'Dropout', 'drop_prob': 0.0},
    init_cfg=None,
    batch_first=False,
    **kwargs,
)

and instantiates the torch.nn.MultiheadAttention module for attention calculation.


mmpretrain.models.utils.attention.MultiheadAttention(
    embed_dims,
    num_heads,
    input_dims=None,
    attn_drop=0.0,
    proj_drop=0.0,
    dropout_layer={'type': 'Dropout', 'drop_prob': 0.0},
    qkv_bias=True,
    qk_scale=None,
    proj_bias=True,
    v_shortcut=False,
    use_layer_scale=False,
    layer_scale_init_value=0.0,
    init_cfg=None,
)

"""


class AttnExportMixin:

	@classmethod
	def install(cls, module_new, replacement_shared):
				
		module_new.extract_multihead = bool(replacement_shared['multihead'])
		module_new.extra_output_storage = replacement_shared['extra_output_storage']

		if replacement_shared['verbose_layers']:
			layer_idx = replacement_shared['layer_idx']
			module_new.extract_message = f'Attn{layer_idx:02d}: '

		else:
			module_new.extract_message = None

		replacement_shared['layer_idx'] += 1

		return module_new
	

	def export_attention(self, attention_weights, outputs=None):
		# print(self.module_path, 'MHA attshape', tuple(attention_weights.shape), 'outshape', tuple(outputs.shape))

		if self.extract_multihead:
			aw = attention_weights.detach()
			num_heads = aw.shape[1]

			for head_idx in range(num_heads):
				self.extra_output_storage[f'{self.module_path}_h{head_idx:02d}'] = aw[:, head_idx]
		else:
			self.extra_output_storage[self.module_path] = attention_weights.detach()


class AttnExport_mmpretrain(MultiheadAttention_mmpretrain, AttnExportMixin):
	"""
	MultiheadAttention_mmpretrain(
		embed_dims,
		num_heads,
		input_dims=None,
		attn_drop=0.0,
		proj_drop=0.0,
		dropout_layer={'type': 'Dropout', 'drop_prob': 0.0},
		qkv_bias=True,
		qk_scale=None,
		proj_bias=True,
		v_shortcut=False,
		use_layer_scale=False,
		layer_scale_init_value=0.0,
		init_cfg=None,
	):
		self.embed_dim = embed_dim


	torch.nn.MultiheadAttention(
		embed_dim,
		num_heads,
		dropout=0.0,
		bias=True,
		add_bias_kv=False,
		add_zero_attn=False,
		kdim=None,
		vdim=None,
		batch_first=False,
		device=None,
		dtype=None,
	
	"""	

	@classmethod
	def from_superclass(cls, module_orig, replacement_shared):
		module_new = cls(
			embed_dims = module_orig.embed_dims,
			num_heads = module_orig.num_heads,
			attn_drop = module_orig.attn_drop,
		)
		obj_copy_attrs(module_new, module_orig)	
		
		# module_new.attention_op = torch.nn.MultiheadAttention(
		# 	embed_dim = module_orig.embed_dims,
		# 	num_heads = module_orig.num_heads,
		# 	dropout = module_orig.attn_drop,
		# )

		return cls.install(module_new, replacement_shared=replacement_shared)


	@staticmethod
	def scaled_dot_product_attention_withweight(
			query,
			key,
			value,
			attn_mask=None,
			dropout_p=0.,
			scale=None,
			is_causal=False):
		scale = scale or query.size(-1)**0.5
		if is_causal and attn_mask is not None:
			attn_mask = torch.ones(
				query.size(-2), key.size(-2), dtype=torch.bool).tril(diagonal=0)
		if attn_mask is not None and attn_mask.dtype == torch.bool:
			attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf'))

		attn_weight = query @ key.transpose(-2, -1) / scale
		if attn_mask is not None:
			attn_weight += attn_mask
		attn_weight = torch.softmax(attn_weight, dim=-1)
		attn_weight = torch.dropout(attn_weight, dropout_p, True)
		return attn_weight @ value, attn_weight

	"""
	Forward is

		def forward(self, x):
			B, N, _ = x.shape
			qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
									self.head_dims).permute(2, 0, 3, 1, 4)
			q, k, v = qkv[0], qkv[1], qkv[2]

			attn_drop = self.attn_drop if self.training else 0.
			x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
			x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

			x = self.proj(x)
			x = self.out_drop(self.gamma1(self.proj_drop(x)))

			if self.v_shortcut:
				x = v.squeeze(1) + x
			return x
	
	"""
	def forward(self, x):
		B, N, _ = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
								self.head_dims).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]

		attn_drop = self.attn_drop if self.training else 0.
		# x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)

		# Altered part!
		# print(self.extract_message, f'q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)} input{tuple(x.shape)}')

		x, attention_weights = self.scaled_dot_product_attention_withweight(q, k, v, dropout_p=attn_drop)

		if not self.extract_multihead:
			# heads are stored on separate channels, average them
			attention_weights = attention_weights.mean(dim=1)

		if self.extract_message:
			print(self.extract_message, f'q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)} x {tuple(x.shape)} atn {tuple(attention_weights.shape)}')
		self.export_attention(attention_weights, x)
		# End altered

		x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

		x = self.proj(x)
		x = self.out_drop(self.gamma1(self.proj_drop(x)))

		if self.v_shortcut:
			x = v.squeeze(1) + x
		return x


class AttnExport_mmcv(MultiheadAttention_mmcv, AttnExportMixin):
	
	@classmethod
	def from_superclass(cls, module_orig, replacement_shared):
		module_new = cls(
			embed_dims = module_orig.embed_dims,
			num_heads = module_orig.num_heads,
			# attn_drop = module_orig.attn_drop,
		)
		obj_copy_attrs(module_new, module_orig)	
		
		return cls.install(module_new, replacement_shared=replacement_shared)


	def forward(self,
			query,
			key=None,
			value=None,
			identity=None,
			query_pos=None,
			key_pos=None,
			attn_mask=None,
			key_padding_mask=None,
			**kwargs):

		if key is None:
			key = query
		if value is None:
			value = key
		if identity is None:
			identity = query
		if key_pos is None:
			if query_pos is not None:
				# use query_pos if key_pos is not available
				if query_pos.shape == key.shape:
					key_pos = query_pos
				else:
					warnings.warn(f'position encoding of key is missing in {self.__class__.__name__}.')

		if query_pos is not None:
			query = query + query_pos
		if key_pos is not None:
			key = key + key_pos

		if self.batch_first:
			query = query.transpose(0, 1)
			key = key.transpose(0, 1)
			value = value.transpose(0, 1)

		out, attention_weights = self.attn(
			query=query,
			key=key,
			value=value,
			attn_mask=attn_mask,
			key_padding_mask=key_padding_mask,
			need_weights=True,
			average_attn_weights = not self.extract_multihead,
		)

		if self.extract_message:
			print(self.extract_message, f'q {tuple(query.shape)} k {tuple(key.shape)} v {tuple(value.shape)} atn {tuple(attention_weights.shape)}')
		
		self.export_attention(attention_weights, out)

		if self.batch_first:
			out = out.transpose(0, 1)

		return identity + self.dropout_layer(self.proj_drop(out))





REPLACEMENT_PAIRS = (
	(MultiheadAttention_mmcv, AttnExport_mmcv),
	(MultiheadAttention_mmpretrain, AttnExport_mmpretrain),
)

def net_install_attn_export(net, verbose_patches=True, verbose_layers=False, class_replacement_pairs=REPLACEMENT_PAIRS):

	# to store attention

	net.extra_output_storage = {}

	replacement_shared = dict(
		extra_output_storage = net.extra_output_storage,
		multihead = False,
		verbose_layers = verbose_layers,
		layer_idx = 0,
	)

	# # inspect number of patches
	# if verbose_patches:
	# 	replace_modules(net, PatchEmbed, PatchEmbed_ViewShapes.from_superclass)

	# capture attention maps
	# replace_modules(net, 
	# 	MultiheadAttention_mmcv, 
	# 	partial(
	# 		AttnExport_mmcv.from_superclass, 
	# 		replacement_shared = replacement_shared,
	# 	),
	# )

	# replace_modules(net, 
	# 	MultiheadAttention_mmpretrain, 
	# 	partial(
	# 		AttnExport_mmpretrain.from_superclass, 
	# 		replacement_shared = replacement_shared,
	# 	),
	# )

	for cls_from, cls_to in class_replacement_pairs:
		replace_modules(net, 
			cls_from, 
			partial(
				cls_to.from_superclass, 
				replacement_shared = replacement_shared,
			),
		)

	# SETR by default infers 768x768 patches in a sliding window manner
	# We overrride this to do a single 768x768 patch in the center
	# See setr_prepare_image in prepare function too!
	# net.test_cfg.mode = 'whole'
	# net.cfg.data.test.pipeline[1].img_scale = (768, 768)

	return net
