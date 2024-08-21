
import re, math
from functools import partial
from pathlib import Path
from easydict import EasyDict
import torch
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
# from mmseg.apis import inference_segmentor
from mmseg.apis import inference_model, MMSegInferencer
from mmseg.models.utils import resize

from math import sqrt


# from mmseg.ops import resize
from .replace import replace_modules, obj_copy_attrs
from .adaptive_threshold import entropy_adaptive_thresholding
from .setr_attention import MultiheadAttention, MultiheadAttention_ExportAttention
from .load import load_mmseg_from_cfg, DEFS_MMSEG_BASE
from ..paths import DIR_MM_CONFIGS, DIR_MM_WEIGHTS
from ..visualization.semantic_overlay import semantic_overlay
from ..evaluation.methods import MethodRegistry


# def calc_entropy_batched_singlech(attn, renorm=False):
# 	# drop class token
# 	dim_B = attn.shape[0]
	
# 	attn_nocls = attn[:, 1:, 1:]
# 	if renorm:
# 		attn_nocls = attn_nocls / attn_nocls.sum(dim=2, keepdims=True)
# 	attn_out_entropy = -(attn_nocls * torch.log(attn_nocls)).sum(dim=2)
# 	attn_out_entropy = attn_out_entropy.reshape(dim_B, 1, 48, 48)
# 	return attn_out_entropy
	
def calc_entropy_batched_singlech(attn, renorm=False):
	"""
	Compared to the segformer version, we
	- strip classtoken
	- output dim is (B, 1, token_side, token_side) not (B, ts, ts)
	"""
	# strip classtoken
	attn_nocls = attn[:, 1:, 1:]
	if renorm:
		attn_nocls = attn_nocls / attn_nocls.sum(dim=2, keepdims=True)

	dim_B, num_tokens, num_attn = attn_nocls.shape
	tokens_on_a_side = round(sqrt(num_tokens))
	out_sh = (dim_B, 1) + (tokens_on_a_side, tokens_on_a_side)

	# attention entropy
	attn_out_entropy = -(attn_nocls * torch.log(attn_nocls)).sum(dim=2) 
	attn_out_entropy = attn_out_entropy.reshape(out_sh)
	return attn_out_entropy



def calc_entropy_diffused_singlech(attn, NUM_ITER=4, BLEND=0.02):
	# drop class token
	dim_B = attn.shape[0]
	
	attn_nocls = attn[0, 1:, 1:]
	# TODO: normalize attention before entropy?
	attn_ent = -(attn_nocls * torch.log(attn_nocls)).sum(dim=1)

	# normalize entropy to always sum to 1 and stop bleeding heat to classtoken
	attn_nocls = attn_nocls / attn_nocls.sum(dim=1, keepdim=True)
	
	state_0 = attn_ent[:, None]
	sublimed = state_0
	sublimed /= sublimed.norm()

	for i in range(NUM_ITER):
		sublimed = (1.-BLEND) * (attn_nocls @ sublimed) + BLEND * state_0
		sublimed /= sublimed.norm()

	return sublimed.reshape(1, 1, 48, 48)


def calc_entropy_batched_multilayer(attn_layers):
	dim_B = attn_layers[0].shape[0]
	dim_C = attn_layers.__len__()

	# drop class token and stack all layers
	attn_nocls_stacked = torch.stack(
		[a[:, 1:, 1:] for a in attn_layers],
		dim=1,
	)
	# entropy calculation maintaining batch and channel dim
	attn_ent = -(attn_nocls_stacked * torch.log(attn_nocls_stacked)).sum(dim=3)
	return attn_ent.reshape((dim_B, dim_C, 48, 48))


import torch.nn.functional as F
# from mmseg.ops import resize

class EncoderDecoder_NoSoftmax(EncoderDecoder):

	def process_fused_logits(self, logits):
		# return F.softmax(logits, dim=1)
		return logits

	# def inference(self, img, img_meta):
	# 	""" 
	# 	Override to not apply softmax 
	# 	This function performs sliding window fusion and undoes any geometric transforms.
	# 	It assumes output is a single array (logits).
	# 	Therefore we store attention entropy instead of logits a single array.
	# 	"""

	# 	assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
	# 		f'Only "slide" or "whole" test mode are supported, but got ' \
	# 		f'{self.test_cfg["mode"]}.'
	# 	ori_shape = batch_img_metas[0]['ori_shape']
	# 	if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
	# 		print_log(
	# 			'Image shapes are different in the batch.',
	# 			logger='current',
	# 			level=logging.WARN)
			
	# 	if self.test_cfg.mode == 'slide':
	# 		seg_logit = self.slide_inference(inputs, batch_img_metas)
	# 	else:
	# 		seg_logit = self.whole_inference(inputs, batch_img_metas)

	# 	return seg_logit
	
	def inference(self, inputs, batch_img_metas):
		"""Inference with slide/whole style.

		Args:
			inputs (Tensor): The input image of shape (N, 3, H, W).
			batch_img_metas (List[dict]): List of image metainfo where each may
				also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
				'ori_shape', 'pad_shape', and 'padding_size'.
				For details on the values of these keys see
				`mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

		Returns:
			Tensor: The segmentation results, seg_logits from model of each
				input image.
		"""
		assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
			f'Only "slide" or "whole" test mode are supported, but got ' \
			f'{self.test_cfg["mode"]}.'
		ori_shape = batch_img_metas[0]['ori_shape']
		if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
			print_log(
				'Image shapes are different in the batch.',
				logger='current',
				level=logging.WARN)
		if self.test_cfg.mode == 'slide':
			seg_logit = self.slide_inference(inputs, batch_img_metas)
		else:
			seg_logit = self.whole_inference(inputs, batch_img_metas)

		return seg_logit

	def simple_test(self, img, img_meta):
		"""

		"""
		return self.inference(img, img_meta)

	def inference_custom(self, image):
		"""
		The output has mysteriously 19 channels - some residual num class from MMseg
		but all of them are equal to our entropy.
		so we take the 0 channel
		"""
		out = inference_model(self, image)
		return out[0, 0]


@MethodRegistry.register_class()
class SETR_AttnEntropyOutput(EncoderDecoder_NoSoftmax):
	"""
	Modify MMseg's network class to output entropy of attention maps from SETR.
	"""

	mode = 'AttnEntropy'
	thr_mode = False
	cls_token_renorm = False

	# CHK_SETR = 'setr_vit-large_pup_8x1_768x768_80k_cityscapes'
	SETR_CTC = 'setr_vit-l_pup_8xb1-80k_cityscapes-768x768'

	"""
	Imagenet-only backbone
		python vit2mmseg.py https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth /cvlabsrc1/cvlab/pytorch_model_zoo/mmlab/vit_large_p16_384-b3be5167.pth
		
	"""

	configs = [
		dict(
			name = 'SETR-AttnEntropy_all',
			mmseg_id = SETR_CTC,
			# variant_name = 'SETR-AttnEntropy_' + '+'.join(map(str, range(24))),
			combination = 'all',
		),
		dict(
			name = 'SETR-AttnEntropyMultihead_allNoSlide',
			mmseg_id = SETR_CTC,
			combination = [1]*24*12,
			multihead = True,
			pipeline_resize = (768, 768),
		),
		dict(
			name = 'SETR-AttnEntropyNm_all',
			mmseg_id = SETR_CTC,
			# variant_name = 'SETR-AttnEntropy_' + '+'.join(map(str, range(24))),
			combination = 'all',
			cls_token_renorm = True,
		),
		dict(
			name = 'SETR-AttnEntropy_optJoint',
			mmseg_id = SETR_CTC,
			combination = [ 1.0041051, 0.53543496, 0.2311193, 0.4245265, 1.5569226, 1.2850144, 0.93282974, 0.7037886, 1.0285404, 0.27811423, 0.54325604, 1.0322673, 0.42179346, 0.24952573, 0.37413964, 0.36526784, -0.3084054, 0.20178239, 0.07209045, 0.1397108, 0.2734738, 0.12465727, 0.30621567, 1.5749772 ], #bias =  [83.26243]
		),
		dict(
			name = 'SETR-AttnEntropy_optObs',
			mmseg_id = SETR_CTC,
			combination = [ 0.94220465, 0.49650165, 0.19390227, 0.40235323, 1.5457597, 0.6486312, 1.0374373, 0.9875229, 1.4211024, 0.31729198, 0.5706229, 1.0544578, -0.013459929, 0.11747559, 0.49340785, 0.293108, 0.26156098, 0.50121343, 0.5442355, 0.5087048, 0.23399672, -0.27153802, 0.10925831, 1.1700678 ], #bias =  [83.02567]
		),
		dict(
			name = 'SETR-AttnEntropy_optLaf',
			mmseg_id = SETR_CTC,
			combination = [ 0.49091464, -0.33114874, 0.66966254, 0.9780063, 1.7613517, 1.3978988, 0.92120016, 0.1837711, 0.99587446, -0.05175174, 0.6498002, 1.1653223, 0.5287516, 0.3163997, 0.49146378, 0.63144904, -0.35066772, 0.10186517, -0.12736036, 0.2763532, 0.24236102, 0.15189055, 0.48054978, 1.6147711 ], #bias =  [83.61014]
		),
		dict(
			name = 'SETR-AttnEntropy_1+2+3+4+5+6+7+8+9+10+11+12+13+23',
			mmseg_id = SETR_CTC,
			combination = '+1+2+3+4+5+6+7+8+9+10+11+12+13+23',
		),
		dict(
			name = 'SETR-AttnEntropy_manual',
			mmseg_id = SETR_CTC,
			combination = '+1+2+3+4+5+6+7+8+9+10+11+12+13+23',
		),

		dict(
			name = 'SETR-AttnEntropy_auto11',
			mmseg_id = SETR_CTC,
			combination = '+0+1+2+3+4+5+6+7+8+9+10+11+12+13+14+23',
		),
		dict(
			name = 'SETR-AttnEntropy_auto12',
			mmseg_id = SETR_CTC,
			combination = '+0+1+2+3+4+5+6+7+8+9+10+11+12+13+23',
		),
		dict(
			name = 'SETR-AttnEntropy_auto12b',
			# mmseg_id = SETR_CTC,
			combination = '+1+2+3+4+5+6+7+8+22+23',
			mmseg_id = SETR_CTC,
		),
		dict(
			name = 'SETR-AttnEntropy_manual_imagenet',
			mmseg_id = SETR_CTC,
			secondary_checkpoint = 'vit_large_p16_384-b3be5167.pth',
			combination = '+1+2+3+4+5+6+7+8+9+10+11+12+13+23',
		),
		dict(
			name = 'SETR-AttnEntropy_manual_noCtc',
			mmseg_id = SETR_CTC,
			combination = '+1+2+3+4+5+6+7+8+9+10+11+12+13+23',
		),
		dict(
			name = 'SETR-AttnEntropyNm_manual',
			mmseg_id = SETR_CTC,
			combination = '+1+2+3+4+5+6+7+8+9+10+11+12+13+23',
			cls_token_renorm = True,
		),
		dict(
			name = 'SETR-AttnEntropy_manualNoSlide',
			mmseg_id = SETR_CTC,
			combination = '+1+2+3+4+5+6+7+8+9+10+11+12+13+23',
			pipeline_resize = (768, 768),
		),

		dict(
			name = 'SETR-AttnEntropyDiffused_1+2+3+4+5+6+7+8+9+10+11+12+13+23',
			mmseg_id = SETR_CTC,
			combination = '+1+2+3+4+5+6+7+8+9+10+11+12+13+23',
		),
		# dict(
			# greedy+11+4+23+6-7+8+5-22+15-16+14-17+20-13+12-21-9+10 - LAF
			# greedy+11+4+23+6-5+8-10+14-22 - obsvalidation
		# ),
		dict(
			name = 'SETR-AttnEntropy_noslide',
			mmseg_id = SETR_CTC,
			pipeline_resize = (768, 768),
			combination = 'all',
			# variant_name = 'SETR-AttnEntropy_' + '+'.join(map(str, range(24))),
		),
		dict(
			name = 'SETR-AttnEntropy_export',
			mmseg_id = SETR_CTC,
			combination = 'all',
			layer_combination_mode = 'export',
		),
		# DPT
		# Config changed to slide mode to achieve square attn map
		# test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(256, 256)))
		dict(
			name = 'DPT-AttnEntropy_all',
			mmseg_id = 'dpt_vit-b16_8xb2-160k_ade20k-512x512',
			combination = 'all',
			num_layers = 12,
		),
		dict(
			name = 'DPT-AttnEntropy_manual',
			mmseg_id = 'dpt_vit-b16_8xb2-160k_ade20k-512x512',
			combination = "+1+2+3+4+5+6+7+8",
			num_layers = 12,
		),
		dict(
			name = 'DPT-AttnEntropy_auto11',
			mmseg_id = 'dpt_vit-b16_8xb2-160k_ade20k-512x512',
			combination = "+0+1+2+3+4+5+6",
			num_layers = 12,
		),
		dict(
			name = 'DPT-AttnEntropy_auto12',
			mmseg_id = 'dpt_vit-b16_8xb2-160k_ade20k-512x512',
			combination = "+0+1+2+3+4+5",
			num_layers = 12,
		),
	
		dict(
			name = 'DPT-AttnEntropy_allNoSlide',
			mmseg_id = 'dpt_vit-b16_8xb2-160k_ade20k-512x512',
			combination = 'all',
			pipeline_resize = (512, 512),
			num_layers = 12,
		),
		dict(
			name = 'DPT-AttnEntropy_manualNoSlide',
			mmseg_id = 'dpt_vit-b16_8xb2-160k_ade20k-512x512',
			combination = "+1+2+3+4+5+6+7+8",
			num_layers = 12,
			pipeline_resize = (512, 512),
		),
		dict(
			name = 'DPT-Imagenet_all',
			checkpoint = 'dpt_imagenet',
			combination = 'all',
			num_layers = 12,
		),
		dict(
			name = 'DPT-Cityonly_all',
			checkpoint = 'dpt_cityonly',
			combination = 'all',
			num_layers = 12,
		),
		dict(
			name = 'DPT-Cityonly_manual',
			checkpoint = 'dpt_cityonly',
			combination = "+1+2+3+4+5+6+7+8",
			num_layers = 12,
		),
		dict(
			name = 'SETR-Cityonly_all',
			checkpoint = 'setr_cityonly',
			# variant_name = 'SETR-AttnEntropy_' + '+'.join(map(str, range(24))),
			combination = 'all',
		),
	]

	@staticmethod
	def parse_combination_str(combination, num_layers=24):
		if combination == 'all':
			return [1] * num_layers
			
		layers = re.findall(r'([+-]\d+)', combination)
		combination = [0] * num_layers

		for layer in layers:
			sign = {'+': 1, '-': -1}[layer[0]]
			lid = int(layer[1:])
			combination[lid] = sign

		return combination

	@staticmethod
	def load_backbone_checkpoint(orig_net, sec_chk_path):

		bg = orig_net.backbone

		sec_chk_path = Path(sec_chk_path)
		sec_chk_path = sec_chk_path if sec_chk_path.is_absolute() else DIR_MM_WEIGHTS / sec_chk_path

		checkpoint = torch.load(sec_chk_path, map_location='cpu')


		# From https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/backbones/vit.py#L247
		
		if 'state_dict' in checkpoint:
			state_dict = checkpoint['state_dict']
		else:
			state_dict = checkpoint

		if 'pos_embed' in state_dict.keys():
			if bg.pos_embed.shape != state_dict['pos_embed'].shape:
				print(f'Resize the pos_embed shape from {state_dict["pos_embed"].shape} to {bg.pos_embed.shape}')
				h, w = bg.img_size
				pos_size = int(
					math.sqrt(state_dict['pos_embed'].shape[1] - 1))
				state_dict['pos_embed'] = bg.resize_pos_embed(
					state_dict['pos_embed'],
					(h // bg.patch_size, w // bg.patch_size),
					(pos_size, pos_size), bg.interpolate_mode)

		bg.load_state_dict(state_dict, False)


	def __init__(self, config):
		self.extra_output_storage = {}

		# don't set net.cfg because that is in use by mmseg
		self.config = config

		# our settings
		combination = config.get('combination')
		if isinstance(combination, str):
			self.combination = self.parse_combination_str(combination)
		elif combination is not None:
			self.combination = combination
		else:
			self.combination = [1] * config.get('num_layers', 24)

		cls_token_renorm = config.get('cls_token_renorm', False)
		if cls_token_renorm:
			self.cls_token_renorm = True

		# load checkpoint
		device = 'cuda:0'

		mmseg_id = config.get('mmseg_id')
		if mmseg_id:
			orig_net = MMSegInferencer(mmseg_id, device=device).model
		else:
			orig_net = load_mmseg_from_cfg(DEFS_MMSEG_BASE[config.checkpoint], device=device)

		# load secondary checkpoint
		sec_chk_path = config.get('secondary_checkpoint', None)
		if sec_chk_path:
			print('Load secondary checkpoint from', sec_chk_path)
			self.load_backbone_checkpoint(orig_net, sec_chk_path)

		obj_copy_attrs(self, orig_net)
		
		# add another channel for entropy
		# net.num_classes = 1

		mode = self.config.get('layer_combination_mode', 'weigted_average')
		
		if mode == 'export':
			# export entropy layers to output
			self.out_channels = self.decode_head.out_channels + config.get('num_layers', 24)
		else:
			self.out_channels = self.decode_head.out_channels + 1
		
		# if verbose_patches:
		# from .setr_attention import PatchEmbed, PatchEmbed_ViewShapes
		# replace_modules(self, PatchEmbed, PatchEmbed_ViewShapes.from_superclass)

		# modify attn layers
		multihead = config.get('multihead')

		# capture attention maps
		replace_modules(self, 
			MultiheadAttention, 
			partial(
				MultiheadAttention_ExportAttention.from_superclass, 
				extra_output_storage = self.extra_output_storage,
				multihead = multihead,
			),
		)

		pipeline_resize = config.get('pipeline_resize', None)
		if pipeline_resize is not None:
			print('pipeline resize', pipeline_resize)
			self.cfg.data.test.pipeline[1].img_scale = tuple(pipeline_resize) # prevent resizing to (1025,1025)


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


		

	# def set_entropy_layers(self, layers_to_sum):
	# 	"""
	# 	Which layers' entropy to sum?
	# 	Defaults to all.
	# 	set_entropy_layers((7,8,9))
	# 	set_entropy_layers('7+8+9')
	# 	"""

	# 	if isinstance(layers_to_sum, str):
	# 		lids_to_convert = layers_to_sum.split('+')
	# 	else:
	# 		lids_to_convert = layers_to_sum
		
	# 	# check that everything is an int and within range
	# 	self.layers_to_sum = tuple(map(int, lids_to_convert))

	# 	for lid in self.layers_to_sum:
	# 		if lid < 0 or lid >= 24:
	# 			raise ValueError(f'Layer {lid} outside of range 0..23')

	# def set_variant(self, set_variant):
	# 	""" SETR-CalcMode_l1+l2+l3 """

	# 	self.name = set_variant

	# 	_, mode_layers = set_variant.split('-')

	# 	mode, layers = mode_layers.split('_')

	# 	if '.' in mode:
	# 		mode, thr = mode.split('.')
	# 		self.thr_mode = True
	# 	else:
	# 		self.thr_mode = False

	# 	if mode in ['AttnEntropy', 'AttnEntropyDiffused']:
	# 		self.mode = mode
	# 	else:
	# 		raise NotImplementedError(f'Mode: {mode}')

	# 	self.set_entropy_layers(layers)


	# @property
	# def name(self):
	# 	return 'SETR-AttnEntropy_' + '+'.join(map(str, sorted(self.layers_to_sum)))

	def __repr__(self):
		return self.name

	@staticmethod
	def remove_classtoken_from_attention(attn):
		return attn[:, 1:, 1:]

	def calc_attnentropy_summed(self, attn_layers):
		# ent_sum = None
		# for lid in self.layers_to_sum:
		# 	ent = calc_entropy_batched_singlech(attn_per_layer[lid])
		# 	if ent_sum is None:
		# 		ent_sum = ent
		# 	else:
		# 		ent_sum += ent

		# output in full resolution?
		# the encode_decode function will resize to image resolution anyway
	
		ent_sum = None
		for (attn, weight) in zip(attn_layers, self.combination):
			if weight != 0:
				ent = calc_entropy_batched_singlech(attn, renorm=self.cls_token_renorm) * weight

				if ent_sum is None:
					ent_sum = ent
				else:
					ent_sum += ent

		return ent_sum

	def calc_attnentropy_diffused(self, attn_per_layer):
		raise NotImplementedError('diffused with combination')

		ent_sum = None

		for lid in self.layers_to_sum:
			ent = calc_entropy_diffused_singlech(attn_per_layer[lid])		
			
			if ent_sum is None:
				ent_sum = ent
			else:
				ent_sum += ent

		# output in full resolution?
		# the encode_decode function will resize to image resolution anyway
		return ent_sum

	def encode_decode(self, img, img_metas):
		"""Encode images with backbone and decode into a semantic segmentation
		map of the same size as input."""

		b_profile = getattr(self, 'b_profile', False)

		with torch.autograd.profiler.profile(enabled=b_profile, use_cuda=True) as prof:
			x = self.extract_feat(img)

		# print('tm backbone', prof.self_cpu_time_total)
		# print(prof)
		if b_profile:
			self.t_backbone = prof.self_cpu_time_total

		out = self._decode_head_forward_test(x, img_metas)

		if b_profile:
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

		# timer_start = torch.cuda.Event()
		# timer_semseg = torch.cuda.Event()
		# timer_entropy = torch.cuda.Event()

		# tm_start = timer_start.record()

		b_profile = getattr(self, 'b_profile', False)

		with torch.autograd.profiler.profile(enabled=b_profile, use_cuda=True) as prof:
			# seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
			seg_logits = self.decode_head.forward(x)

		
		if b_profile:
			self.t_decode = prof.self_cpu_time_total
		# print(prof)

		# tm_semseg = timer_semseg.record()

		# attention_maps = dict(self.extra_output_storage)
		with torch.autograd.profiler.profile(enabled=b_profile, use_cuda=True) as prof:

			attn_per_layer = list(self.extra_output_storage.values())

			mode = self.config.get('layer_combination_mode', 'weigted_average')

			if mode == 'weigted_average':
				attn_ent = self.calc_attnentropy_summed(attn_per_layer)
			elif mode == 'diffused':
				attn_ent = self.calc_attnentropy_diffused(attn_per_layer)
			elif mode == 'export':
				attn_ent = torch.cat([
					calc_entropy_batched_singlech(attn, renorm=self.cls_token_renorm)
					for attn in attn_per_layer
				], dim=1)
				
			if self.thr_mode:
				score = entropy_adaptive_thresholding(-attn_ent[0, 0].cpu().numpy())
				score = -torch.from_numpy(score).float().to(attn_ent.device)[None, None]

			else:
				score = attn_ent

			score = resize(
				score,
				size = seg_logits.shape[2:4],
				mode = 'bilinear',
				align_corners = self.align_corners,
				warning = False,
			)

		if b_profile:
			self.t_entropy = prof.self_cpu_time_total

		# tm_entropy = timer_entropy.record()

		# torch.cuda.synchronize()

		# t_semseg = timer_start.elapsed_time(timer_semseg)
		# t_entropy = 1
		# t_entropy = timer_semseg.elapsed_time(timer_entropy)

		return torch.cat([score, seg_logits], dim=1)

	def process_fused_logits(self, logits):
		# return super().process_fused_logits(logits)
		return logits

	def simple_test(self, img, img_meta, rescale=True):
		"""

		"""
		return self.inference(img, img_meta, rescale)
		
	def inference_custom_vis(self, image):
		mode = self.config.get('layer_combination_mode', 'weigted_average')
		if mode == 'export':
			raise NotImplementedError()

		with torch.no_grad():
			entropy_and_logit = inference_model(self, image)

			attn_per_layer = list(self.extra_output_storage.values())
			attn_layers = attn_per_layer
			attn_layer_names = list(self.extra_output_storage.keys())
			self.extra_output_storage.clear()

			entropy = entropy_and_logit[:, 0]
			seg_class = entropy_and_logit[:, 1:].argmax(dim=1)

			return EasyDict(
				anomaly_p = -entropy[0],
				seg_class = seg_class[0],
				seg_overlay = semantic_overlay(image, seg_class[0], self.PALETTE),
				attn_layers = attn_layers,
				attn_layer_names = attn_layer_names,
			)

	def inference_custom(self, image):
		with torch.no_grad():
			# entropy_and_logit = inference_model(self, image)

			out = inference_model(self, image)
			entropy_and_logit = out.seg_logits.data
			self.extra_output_storage.clear()

			# return out
			# print(entropy_and_logit)

			mode = self.config.get('layer_combination_mode', 'weigted_average')

			if mode == 'export':
				num_layers = self.config.get('num_layers', 24)
				entropy = entropy_and_logit[:, :num_layers]
				seg_class = entropy_and_logit[:, num_layers:].argmax(dim=1)
			
				return EasyDict(
					anomaly_p = -torch.sum(entropy, dim=1)[0],
					entropy_layers = entropy[0],
					seg_class = seg_class[0],
					# seg_overlay = semantic_overlay(image, seg_class[0], self.PALETTE)
				)

			
			else:
				entropy = entropy_and_logit[:, 0]
				seg_class = entropy_and_logit[:, 1:].argmax(dim=1)

				return EasyDict(
					anomaly_p = -entropy[0],
					seg_class = seg_class[0],
					out_raw = entropy_and_logit,
					# seg_overlay = semantic_overlay(image, seg_class[0], self.PALETTE)
				)


	# @classmethod
	# def from_superclass(cls, orig_net, config=None):
	# 	net = cls()
	# 	obj_copy_attrs(net, orig_net)
		
	# 	# add another channel for entropy
	# 	# net.num_classes = 1
	# 	net.out_channels = net.decode_head.out_channels + 1

	# 	multihead = config.get('multihead') if config is not None else False

	# 	# capture attention maps
	# 	replace_modules(net, 
	# 		MultiheadAttention, 
	# 		partial(
	# 			MultiheadAttention_ExportAttention.from_superclass, 
	# 			extra_output_storage = net.extra_output_storage,
	# 			multihead = multihead,
	# 		),
	# 	)

	# 	return net




# @MethodRegistry.register_class()
# class SETR_Attentropy:
# 	CHK_SETR = 'setr_vit-large_pup_8x1_768x768_80k_cityscapes'

# 	configs = [
# 		dict(
# 			name = 'SETR-AttnEntropy_all',
# 			mmseg_id = SETR_CTC,
# 			# variant_name = 'SETR-AttnEntropy_' + '+'.join(map(str, range(24))),
# 			combination = 'all',
# 		),
# 		dict(
# 			name = 'SETR-AttnEntropyNm_all',
# 			mmseg_id = SETR_CTC,
# 			# variant_name = 'SETR-AttnEntropy_' + '+'.join(map(str, range(24))),
# 			combination = 'all',
# 			cls_token_renorm = True,
# 		),
# 		dict(
# 			name = 'SETR-AttnEntropy_optJoint',
# 			mmseg_id = SETR_CTC,
# 			combination = [ 1.0041051, 0.53543496, 0.2311193, 0.4245265, 1.5569226, 1.2850144, 0.93282974, 0.7037886, 1.0285404, 0.27811423, 0.54325604, 1.0322673, 0.42179346, 0.24952573, 0.37413964, 0.36526784, -0.3084054, 0.20178239, 0.07209045, 0.1397108, 0.2734738, 0.12465727, 0.30621567, 1.5749772 ], #bias =  [83.26243]
# 		),
# 		dict(
# 			name = 'SETR-AttnEntropy_optObs',
# 			mmseg_id = SETR_CTC,
# 			combination = [ 0.94220465, 0.49650165, 0.19390227, 0.40235323, 1.5457597, 0.6486312, 1.0374373, 0.9875229, 1.4211024, 0.31729198, 0.5706229, 1.0544578, -0.013459929, 0.11747559, 0.49340785, 0.293108, 0.26156098, 0.50121343, 0.5442355, 0.5087048, 0.23399672, -0.27153802, 0.10925831, 1.1700678 ], #bias =  [83.02567]
# 		),
# 		dict(
# 			name = 'SETR-AttnEntropy_optLaf',
# 			mmseg_id = SETR_CTC,
# 			combination = [ 0.49091464, -0.33114874, 0.66966254, 0.9780063, 1.7613517, 1.3978988, 0.92120016, 0.1837711, 0.99587446, -0.05175174, 0.6498002, 1.1653223, 0.5287516, 0.3163997, 0.49146378, 0.63144904, -0.35066772, 0.10186517, -0.12736036, 0.2763532, 0.24236102, 0.15189055, 0.48054978, 1.6147711 ], #bias =  [83.61014]
# 		),
# 		dict(
# 			name = 'SETR-AttnEntropy_1+2+3+4+5+6+7+8+9+10+11+12+13+23',
# 			mmseg_id = SETR_CTC,
# 			combination = '+1+2+3+4+5+6+7+8+9+10+11+12+13+23',
# 		),
# 		dict(
# 			name = 'SETR-AttnEntropy_manual',
# 			mmseg_id = SETR_CTC,
# 			combination = '+1+2+3+4+5+6+7+8+9+10+11+12+13+23',
# 		),
# 		dict(
# 			name = 'SETR-AttnEntropyNm_manual',
# 			mmseg_id = SETR_CTC,
# 			combination = '+1+2+3+4+5+6+7+8+9+10+11+12+13+23',
# 			cls_token_renorm = True,
# 		),
# 		dict(
# 			name = 'SETR-AttnEntropy_manualNoSlide',
# 			mmseg_id = SETR_CTC,
# 			combination = '+1+2+3+4+5+6+7+8+9+10+11+12+13+23',
# 			pipeline_resize = (768, 768),
# 		),
# 		dict(
# 			name = 'SETR-AttnEntropyDiffused_1+2+3+4+5+6+7+8+9+10+11+12+13+23',
# 			mmseg_id = SETR_CTC,
# 			combination = '+1+2+3+4+5+6+7+8+9+10+11+12+13+23',
# 		),
# 		# dict(
# 			# greedy+11+4+23+6-7+8+5-22+15-16+14-17+20-13+12-21-9+10 - LAF
# 			# greedy+11+4+23+6-5+8-10+14-22 - obsvalidation
# 		# ),
# 		dict(
# 			name = 'SETR-AttnEntropy_noslide',
# 			mmseg_id = SETR_CTC,
# 			pipeline_resize = (768, 768),
# 			combination = 'all',
# 			# variant_name = 'SETR-AttnEntropy_' + '+'.join(map(str, range(24))),
# 		),
# 	]

# 	@staticmethod
# 	def parse_combination_str(combination, num_layers=24):
# 		if combination == 'all':
# 			return [1] * num_layers
			
# 		layers = re.findall(r'([+-]\d+)', combination)
# 		combination = [0] * num_layers

# 		for layer in layers:
# 			sign = {'+': 1, '-': -1}[layer[0]]
# 			lid = int(layer[1:])
# 			combination[lid] = sign

# 		return combination

# 	def __new__(cls, config):
# 		if isinstance(config.combination, str):
# 			combination = cls.parse_combination_str(config.combination)
# 		else:
# 			combination = config.combination

# 		device = 'cuda:0'
# 		net_segf_base = load_mmseg_from_cfg(DEFS_MMSEG_BASE[config.checkpoint], device=device)
# 		net = SETR_AttnEntropyOutput.from_superclass(net_segf_base)
# 		net.config = config
# 		net.combination = combination

# 		pipeline_resize = config.get('pipeline_resize', None)
# 		if pipeline_resize is not None:
# 			raise NotImplementedError('pipeline_resize')
# 			print('pipeline resize', pipeline_resize)
# 			net.cfg.data.test.pipeline[1].img_scale = tuple(pipeline_resize) # prevent resizing to (1025,1025)

# 		cls_token_renorm = config.get('cls_token_renorm', False)
# 		if cls_token_renorm:
# 			net.cls_token_renorm = True

# 		# don't set net.cfg because that is in use by mmseg
# 		return net
	

"""

fr = dset_obs[5]

net_setr.set_entropy_layers(range(24))
res = inference_segmentor(net_setr, fr.image)
print(res.shape)

# The output has mysteriously 19 channels - some residual num class from MMseg
# but all of them are equal to our entropy.
# so we take the 0 channel

e1 = res[0, 0].cpu().numpy()

net_setr.set_entropy_layers(list(range(1, 14))+[23])
e2 = inference_segmentor(net_setr, fr.image)[0, 0].cpu().numpy()



show([e1, e2])

"""