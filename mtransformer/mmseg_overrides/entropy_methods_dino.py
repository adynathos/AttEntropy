
from pathlib import Path
from easydict import EasyDict
import torch
from kornia.utils import image_to_tensor
from kornia.geometry import pyrdown
import mmpretrain

from ..evaluation.methods import MethodRegistry
from .extract_attention import net_install_attn_export, obj_copy_attrs, AttnExportMixin
from .entropy_methods_mmpretrain import AttEntropy_SingleImage

# from LOST.networks import get_model

nn = torch.nn


class DinoAttention_AttnExport(torch.nn.Module, AttnExportMixin):
	def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = qk_scale or head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
	
		if self.extract_message:
			print(self.extract_message, f'q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)} atn {tuple(attn.shape)}')
		
		if not self.extract_multihead:
			# heads are stored on separate channels, average them
			self.export_attention(attn.detach().mean(dim=1))
		else:
			self.export_attention(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x, attn


	@classmethod
	def from_superclass(cls, module_orig, replacement_shared):
		module_new = cls(
			dim = 32,
			num_heads = module_orig.num_heads,
			qk_scale = module_orig.scale,
		)
		obj_copy_attrs(module_new, module_orig)	
		
		return cls.install(module_new, replacement_shared=replacement_shared)


from .entropy_methods_mmpretrain import extend_with_nocombination_configs

@MethodRegistry.register_class()
class AttEntropy_SingleImage_Dino(AttEntropy_SingleImage):
	
	configs = extend_with_nocombination_configs([
		# ViT with imagenet pretraining
		# https://github.com/open-mmlab/mmpretrain/tree/main/configs/vision_transformer
		dict(
			name = 'VitBase-Dino',
			dino_id = 'dino_vitb16',
			combination = '+2+3+9',
		),
	])


	def __init__(self, cfg):
		self.config = cfg
		self.device = torch.device('cuda:0')

		net = torch.hub.load('facebookresearch/dino:main', cfg.dino_id).to(self.device)

		DinoAttentionClass = net.blocks[0].attn.__class__

		net = net_install_attn_export(
			net, 
			verbose_layers=False, 
			class_replacement_pairs=(
				(DinoAttentionClass, DinoAttention_AttnExport),
			),
		)

		self.net = net

	
