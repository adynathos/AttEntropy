from functools import partial

import torch
import cv2
import numpy as np
from mmcv.cnn.bricks.transformer import MultiheadAttention
from show_image import adapt_img_data

from .inference_with_logits import EncoderDecoder_LogitOutput
from .replace import replace_modules, obj_copy_attrs, PatchEmbed, PatchEmbed_ViewShapes

setr_cfg = dict(
	cfg = 'setr_vit-large_pup_8x1_768x768_80k_cityscapes',
	weight_file = 'setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth',
)


class MultiheadAttention_ExportAttention(MultiheadAttention):
	
	def export_attention(self, attention_weights, outputs=None):
		# print(self.module_path, 'MHA attshape', tuple(attention_weights.shape), 'outshape', tuple(outputs.shape))

		if self.multihead:
			aw = attention_weights.detach()
			num_heads = aw.shape[1]

			for head_idx in range(num_heads):
				self.extra_output_storage[f'{self.module_path}_h{head_idx:02d}'] = aw[:, head_idx]
		else:
			self.extra_output_storage[self.module_path] = attention_weights.detach()

	@classmethod
	def from_superclass(cls, module_orig, extra_output_storage, multihead=False):
		module_new = cls(
			embed_dims = module_orig.embed_dims,
			num_heads = module_orig.num_heads,
		)
		
		obj_copy_attrs(module_new, module_orig)	
		module_new.extra_output_storage = extra_output_storage
		module_new.multihead = bool(multihead)

		return module_new
	
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
			average_attn_weights = not self.multihead,
		)
		
		self.export_attention(attention_weights, out)

		if self.batch_first:
			out = out.transpose(0, 1)

		return identity + self.dropout_layer(self.proj_drop(out))


def setr_prepare_image(image_np):
	return cv2.resize(image_np[:, 512:1536], (768, 768))

class EncoderDecoder_SetrAttn(EncoderDecoder_LogitOutput):
	def inference_custom(self, image_np):
		image_resized = setr_prepare_image(image_np)
		out = super().inference_custom(image_resized)
		out.image_resized = image_resized
		out.attention_maps = dict(self.extra_output_storage)
		self.extra_output_storage.clear()
		return out


def net_override_setr_attention(net, verbose_patches=True):

	# output mode: logits and postprocessing
	net = EncoderDecoder_SetrAttn.from_superclass(net)

	# to store attention
	net.extra_output_storage = {}

	# inspect number of patches
	if verbose_patches:
		replace_modules(net, PatchEmbed, PatchEmbed_ViewShapes.from_superclass)

	# capture attention maps
	replace_modules(net, 
		MultiheadAttention, 
		partial(
			MultiheadAttention_ExportAttention.from_superclass, 
			extra_output_storage = net.extra_output_storage,
		),
	)

	# SETR by default infers 768x768 patches in a sliding window manner
	# We overrride this to do a single 768x768 patch in the center
	# See setr_prepare_image in prepare function too!
	net.test_cfg.mode = 'whole'
	net.cfg.data.test.pipeline[1].img_scale = (768, 768)

	return net

