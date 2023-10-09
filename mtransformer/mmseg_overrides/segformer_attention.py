

from functools import partial
import re
from easydict import EasyDict
import torch
from math import sqrt
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.backbones.mit import EfficientMultiheadAttention
from mmseg.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw
from mmseg.apis import inference_segmentor
from mmseg.ops import resize

from .replace import replace_modules, obj_copy_attrs
from .inference_with_logits import softmax_entropy, logits_to_logsumexp
from .setr_entropy import EncoderDecoder_NoSoftmax
from .load import load_mmseg_from_cfg, DEFS_MMSEG_BASE
from ..visualization.semantic_overlay import semantic_overlay
from ..evaluation.methods import MethodRegistry

def print_unknown(x):
	if isinstance(x, tuple):
		return '(' + ', '.join(map(print_unknown, x)) + ')'
	elif isinstance(x, torch.Tensor):
		return f'{x.dtype} x {tuple(x.shape)}'
	else:
		return str(x)


class ProfilerMixin:
	def __init__(self):
		self.ts_backbone = []
		self.ts_decode = []
		self.ts_entropy = []
		
	def profiler_enable(self, b_enable=True):
		self.b_profile = b_enable

	def profiler_print(self, skip=2):
		import numpy as np
		to_ms = 1e-3
		skip = min(self.ts_backbone.__len__()-1, skip)
		ts_backbone = np.array(self.ts_backbone)[skip:] * to_ms
		ts_decode = np.array(self.ts_decode)[skip:] * to_ms
		ts_entropy = np.array(self.ts_entropy)[skip:] * to_ms

		ts_relative = ts_entropy / (ts_backbone + ts_decode)
		print(f"""Profiler for {self.config.name}: {ts_relative.__len__()} samples:
	backbone: {np.mean(ts_backbone)} +- {np.std(ts_backbone)} ms
	decode: {np.mean(ts_decode)} +- {np.std(ts_decode)} ms
	entropy: {np.mean(ts_entropy)} +- {np.std(ts_entropy)} ms
	relative: [{np.mean(ts_relative)*100} +- {np.std(ts_relative)*100}]%
""")

class EfficientMultiheadAttention_ExportAttention(EfficientMultiheadAttention):
	
	def export_attention(self, attention_weights, outputs=None):
		# print(self.module_path, 'MHA attshape', tuple(attention_weights.shape), 'outshape', tuple(outputs.shape))
		self.extra_output_storage[self.module_path] = attention_weights.detach()

	@classmethod
	def from_superclass(cls, module_orig, extra_output_storage):
		module_new = cls(
			embed_dims = module_orig.embed_dims,
			num_heads = module_orig.num_heads,
		)
		
		obj_copy_attrs(module_new, module_orig)	
		module_new.extra_output_storage = extra_output_storage
		return module_new
	
	def forward(self, x, hw_shape, identity=None):
		# print(self.module_path, print_unknown(x), 'hw_shape', hw_shape, 'ratio', self.sr_ratio)

		# return super().forward(x, hw_shape, identity=identity)

		x_q = x
		if self.sr_ratio > 1:
			x_kv = nlc_to_nchw(x, hw_shape)
			x_kv = self.sr(x_kv)
			x_kv = nchw_to_nlc(x_kv)
			x_kv = self.norm(x_kv)
		else:
			x_kv = x



		if identity is None:
			identity = x_q

		# Because the dataflow('key', 'query', 'value') of
		# ``torch.nn.MultiheadAttention`` is (num_query, batch,
		# embed_dims), We should adjust the shape of dataflow from
		# batch_first (batch, num_query, embed_dims) to num_query_first
		# (num_query ,batch, embed_dims), and recover ``attn_output``
		# from num_query_first to batch_first.
		if self.batch_first:
			x_q = x_q.transpose(0, 1)
			x_kv = x_kv.transpose(0, 1)

		# TODO Could be achieved by a hook on attention modules
		# print(f'	query {tuple(x_q.shape)} key {tuple(x_kv.shape)} value {tuple(x_kv.shape)}')
		# out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]
		out, attention = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=True)
		# print('	res', print_unknown(out))
		# out = out[0]

		self.export_attention(attention)

		if self.batch_first:
			out = out.transpose(0, 1)

		return identity + self.dropout_layer(self.proj_drop(out))

		

def attn_entropy(attn):
	dim_B, num_tokens, num_attn = attn.shape
	tokens_on_a_side = round(sqrt(num_tokens))
	out_sh = (dim_B, ) + (tokens_on_a_side, tokens_on_a_side)

	# attention entropy
	attn_out_entropy = -(attn * torch.log(attn)).sum(dim=2) 
	attn_out_entropy = attn_out_entropy.reshape(out_sh)
	return attn_out_entropy


def aggregate_layers_into_same_size(aes, sum=True):
	"""
	@param sum: If True sum, else concat.
	"""
	
	max_side = max(ae.shape[1] for ae in aes)
	
	for ae in aes:
		if ae.shape[1] != ae.shape[2]:
			raise ValueError(f'Attention entropy size is not square but {tuple(ae.shape)}')
	
	def resize(ae):
		if ae.shape[1] < max_side:
			return torch.nn.functional.interpolate(ae[None], (max_side, max_side))[0]
		else:
			return ae
	
	if sum:
		ae_sum = resize(aes[0].clone())
		for ae in aes[1:]:
			ae_sum += resize(ae)

		return ae_sum

	else:
		return torch.cat([
			resize(ae) for ae in aes
		], dim=1)
	

from mtransformer.evaluation.island import attn_mask_to_road


@MethodRegistry.register_class()
class Segformer_Attentropy(EncoderDecoder_NoSoftmax, ProfilerMixin):

	CHK_SEGF = 'segformer_mit-b3_8x1_1024x1024_160k_cityscapes'
	NUM_LAYERS = 28

	b_profile = False

	configs = [
		dict(
			name = 'Segformer-AttnEnt_sumAll',
			checkpoint = CHK_SEGF,
			combination = 'all',

			pipeline_resize = (1024, 1024),
		),
		dict(
			name = 'Segformer-AttnEnt_manual',
			checkpoint = CHK_SEGF,
			combination = "+1+2+#+4+6+7+8+9+10+12+13+15+16+19+21+23+25",
		),
		dict(
			name = 'Segformer-AttnEnt_auto11',
			checkpoint = CHK_SEGF,
			combination = "+0+1+2+3+4+6+7+8+10+11+12+13+15+25+26+27",
		),
		dict(
			name = 'Segformer-AttnEnt_auto12',
			checkpoint = CHK_SEGF,
			combination = "0+2+4+6+7+11+15+25+27",
		),
		dict(
			name = 'Segformer-AttnEnt_optJoint',
			checkpoint = CHK_SEGF,
			combination = [ 0.33730167, 0.11616325, 0.89033437, -0.48228914, 0.47086918, -0.93136835, 2.4063137, 1.4425184, 0.37839597, 0.21083927, 0.51211005, 0.073668666, 0.68472016, 0.06808761, 0.0028231647, 1.8246439, -0.10754192, 0.096895784, -0.7653303, 1.1297168, 0.50567526, 1.7529994, 0.7312622, 1.610035, 1.299432, 1.3423423, -1.2445847, -0.07437456 ], #bias =  [86.84621]
		),
		dict(
			name = 'Segformer-AttnEnt_optObs',
			checkpoint = CHK_SEGF,
			combination = [ 0.6083089, 0.15593, 0.85139406, 0.100575596, -0.62873065, -1.3786778, 1.3501173, 1.3373368, 0.22611064, -0.012760008, 0.4221683, -0.24612471, 0.5519549, 0.14429426, 0.20622164, 2.0551383, 0.034085914, 0.7040446, 0.52296174, 1.765389, 0.45049816, 2.4024172, 0.8139978, 1.2778698, 1.0075419, 0.82226515, -1.34623, -0.16709495 ], #bias =  [87.13472]
		),
		dict(
			name = 'Segformer-AttnEnt_optLaf',
			checkpoint = CHK_SEGF,
			combination = [ 0.16748449, 0.5898432, 0.44531915, 0.27853543, 1.5882809, -0.098556854, 3.844708, 0.6564607, 0.8469239, 0.54244345, 0.7001611, 0.49135143, 0.63093483, -0.04046512, -0.008938782, 0.8379052, -0.51683015, 0.18034464, -1.3662316, 0.49477276, 0.10997535, 1.122988, 0.40890017, 1.187416, 0.8167618, 1.8650105, -1.2977643, -0.032748315 ], #bias =  [86.56312]
		),
		dict(
			name = 'Segformer-AttnEnt_greedy1',
			checkpoint = CHK_SEGF,
			combination = '+19+7+21+2+15-5-14+6-4+3-26-22+10+24-17-9-8+12-11',
		),
		dict(
			name = 'Segformer-AttnEntRM_sumAll',
			checkpoint = CHK_SEGF,
			combination = 'all',
			attn_road_masking_on = True,
		),
		dict(
			name = 'Segformer-AttnEntRM_greedyLt',
			checkpoint = CHK_SEGF,
			combination = '+7+6+16+1+15-27+11-14-24+22-23+26-25+13-17+19-18+12-8-21+9-10+4-5+20', # Laftrain
			# combination = '+7-14+15-25+4-22+21-23+12-8+16-18+19+6-26+27-24+9-20+13-10+2-1', # obsval
			attn_road_masking_on = True,
		),
		dict(
			name = 'Segformer-AttnEntRM_greedyObv',
			checkpoint = CHK_SEGF,
			combination = '+7-14+15-25+4-22+21-23+12-8+16-18+19+6-26+27-24+9-20+13-10+2-1', # obsval
			attn_road_masking_on = True,
		),
	]


	b_profile = False
	attn_road_masking_on = False

	def __init__(self, config):
		self.config = config

		# to store attention
		self.extra_output_storage = {}
		self.combination = [1] * self.NUM_LAYERS

		if isinstance(config.combination, str):
			combination = self.parse_combination_str(config.combination)
		else:
			combination = config.combination

		device = 'cuda:0'
		net_segf_base = load_mmseg_from_cfg(DEFS_MMSEG_BASE[config.checkpoint], device=device)
		# net = Segformer_Custom.from_superclass(net_segf_base)
		self.combination = combination
		self.attn_road_masking_on = config.get('attn_road_masking_on', False)

		obj_copy_attrs(self, net_segf_base)
		
		mode = self.config.get('layer_combination_mode', 'weigted_average')
		
		if mode == 'export':
			# export entropy layers to output
			self.out_channels = self.decode_head.out_channels + self.NUM_LAYERS
		else:
			# add another channel for entropy
			self.out_channels = self.decode_head.out_channels + 1

		
		# capture attention maps
		replace_modules(self, 
			EfficientMultiheadAttention, 
			partial(
				EfficientMultiheadAttention_ExportAttention.from_superclass, 
				extra_output_storage = self.extra_output_storage,
			),
			verbose=True,
		)

		self.ts_backbone = []
		self.ts_decode = []
		self.ts_entropy = []

	@staticmethod
	def remove_classtoken_from_attention(attn):
		return attn

	@staticmethod
	def parse_combination_str(combination, num_layers=28):
		if combination == 'all':
			return [1] * num_layers
			
		layers = re.findall(r'([+-]\d+)', combination)
		combination = [0] * num_layers

		for layer in layers:
			sign = {'+': 1, '-': -1}[layer[0]]
			lid = int(layer[1:])
			combination[lid] = sign

		return combination


	def encode_decode(self, img, img_metas):
		"""Encode images with backbone and decode into a semantic segmentation
		map of the same size as input."""

		with torch.autograd.profiler.profile(enabled=self.b_profile, use_cuda=True) as prof:
			x = self.extract_feat(img)

		# print('tm backbone', prof.self_cpu_time_total)
		# print(prof)
		if self.b_profile:
			self.t_backbone = prof.self_cpu_time_total

		out = self._decode_head_forward_test(x, img_metas)

		if self.b_profile:
			self.ts_backbone.append(self.t_backbone)
			self.ts_decode.append(self.t_decode)
			self.ts_entropy.append(self.t_entropy)
	
		# print('t_backbone', self.t_backbone, 't_decode', self.t_decode, 't_entropy', self.t_entropy, 
		# 	'relative', self.t_entropy / (self.t_backbone + self.t_decode),
		# )


		out = resize(
			input=out,
			size=img.shape[2:],
			mode='bilinear',
			align_corners=self.align_corners)
		return out

	def _decode_head_forward_test(self, x, img_metas):
		"""
		Call stack in inference is:
		- simple_test
		- inference
		- encode_decode
		- _decode_head_forward_test
		"""

		with torch.autograd.profiler.profile(enabled=self.b_profile, use_cuda=True) as prof:
			seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
		
		if self.b_profile:
			self.t_decode = prof.self_cpu_time_total

		with torch.autograd.profiler.profile(enabled=self.b_profile, use_cuda=True) as prof:
		

			attn_layers = list(self.extra_output_storage.values())

			if self.attn_road_masking_on:
				seg_class = seg_logits.argmax(dim=1)
				attn_layers = attn_mask_to_road(seg_class[0], attn_layers).attn_maps		

			attn_ent_layers = [
				- attn_entropy(attn) * weight
				for (attn, weight) in 
				zip(attn_layers, self.combination) if weight != 0
			]

			score = aggregate_layers_into_same_size(attn_ent_layers)[:, None]

			score = resize(
				score,
				size = seg_logits.shape[2:4],
				mode = 'bilinear',
				align_corners = self.align_corners,
				warning = False,
			)

		if self.b_profile:
			self.t_entropy = prof.self_cpu_time_total
		
		return torch.cat([score, seg_logits], dim=1)

	def process_fused_logits(self, logits):
		# return super().process_fused_logits(logits)
		return logits


	def inference_custom(self, image):
		out = EasyDict()

		with torch.no_grad():
			entropy_and_logit = inference_segmentor(self, image)
			self.extra_output_storage.clear()

			entropy = entropy_and_logit[:, 0]
			seg_class = entropy_and_logit[:, 1:].argmax(dim=1)

			out.anomaly_p = entropy[0]
			out.seg_class = seg_class[0]
			out.seg_overlay = semantic_overlay(image, out.seg_class, self.PALETTE)

		return out

	def inference_custom_vis(self, image):
		out = EasyDict()

		with torch.no_grad():
			seg_logits = inference_segmentor(self, image)[0] # no batch dim
			seg_class = torch.argmax(seg_logits, dim=0)
			out.seg_overlay = semantic_overlay(image, seg_class, self.PALETTE)
			out.seg_class = seg_class.cpu()

			seg_softmax = torch.nn.functional.softmax(seg_logits, dim=0) # no batch dim
			out.seg_entropy = softmax_entropy(seg_softmax)
			out.seg_logsumexp = logits_to_logsumexp(seg_logits)

			# attention
			attn_per_layer = list(self.extra_output_storage.values())
			out.attn_layers = attn_per_layer
			out.attn_layer_names = list(self.extra_output_storage.keys())
			self.extra_output_storage.clear()
			out.attn_maps = attn_per_layer
			out.attn_entropy = [attn_entropy(a)[0].cpu() for a in attn_per_layer]

		return out

	@classmethod
	def from_superclass(cls, orig_net):
		net = cls()
		obj_copy_attrs(net, orig_net)
		
		# add another channel for entropy
		net.out_channels = net.decode_head.out_channels + 1

		# capture attention maps
		replace_modules(net, 
			EfficientMultiheadAttention, 
			partial(
				EfficientMultiheadAttention_ExportAttention.from_superclass, 
				extra_output_storage = net.extra_output_storage,
			),
			verbose=True,
		)

		return net

