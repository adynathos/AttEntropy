
from pathlib import Path
from easydict import EasyDict
import torch
from kornia.utils import image_to_tensor
from kornia.geometry import pyrdown
import mmpretrain

from ..evaluation.methods import MethodRegistry
from .extract_attention import net_install_attn_export, AttnExportMixin
from .extract_entropy import process_image_and_extract_attentropy, process_image_and_extract_score


class AttEntropy_SingleImage:

	@property
	def name(self):
		return self.config.name

	def prepare_image(self, image):
		"""
		@param image: np image from dataset
		@return: torch image on the correct device
		"""
		img_tr = image_to_tensor(image).float() * (1./255.)
		img_tr = img_tr[None].to(self.device)
		return img_tr

	def inference_layers(self, image):
		"""
		Process a frame and return the attnetropy layers to allow for layer analysis.
		"""

		with torch.no_grad():
			img_tr = self.prepare_image(image)
			entropies = process_image_and_extract_attentropy(self.net, img_tr)
			return entropies


	def inference_custom(self, image):
		"""
		Process a frame and calculate a weighted average of entropy layers
		to produce anomaly score for evaluation.
		"""

		with torch.no_grad():

			img_tr = self.prepare_image(image)
			# entropies = process_image_and_extract_attentropy(self.net, img_tr)

			combination_str = self.config.get('combination', 'all')

			score = process_image_and_extract_score(self.net, img_tr, combination=combination_str)

			# weight
			return EasyDict(
				anomaly_p = score,
			)

	
def extend_with_nocombination_configs(configs):

	configs_extended = []

	for c in configs:
		c_nocomb = dict(c)
		if "combination" in c_nocomb:
			del c_nocomb["combination"]
		c_nocomb["name"] += "_CAll"

		configs_extended += [c, c_nocomb]

	return configs_extended


@MethodRegistry.register_class()
class AttEntropy_SingleImage_MMPretrain(AttEntropy_SingleImage):
	
	configs = extend_with_nocombination_configs([
		# ViT with imagenet pretraining
		# https://github.com/open-mmlab/mmpretrain/tree/main/configs/vision_transformer
		dict(
			name = 'VitBase-ImgNet',
			mmpretrain_id = 'vit-base-p16_in21k-pre_3rdparty_in1k-384px',
			combination = '+1+2+3+4+5+6+7+8+9+10',
		),
		dict(
			name = 'VitLarge-ImgNet',
			mmpretrain_id = 'vit-large-p16_in21k-pre_3rdparty_in1k-384px',
			combination = '+2+3+5+6+7+8+9+10+11+12+13+14+15+16+17+21+22+23',
		),

		# MAE masked self-supervised
		# https://github.com/open-mmlab/mmpretrain/tree/main/configs/mae
		dict(
			name = 'VitBase-MAE-unsup',
			mmpretrain_id = 'mae_vit-base-p16_8xb512-amp-coslr-1600e_in1k',
			combination = '+5+7+9+10',
		),
		dict(
			name = 'VitLarge-MAE-unsup',
			mmpretrain_id = 'mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k',
			combination = '+4+5+9+10',
		),

		dict(
			name = 'VitBase-MAE-ImageNetCls',
			mmpretrain_id = 'vit-base-p16_mae-1600e-pre_8xb128-coslr-100e_in1k',
			combination = '+2+3+5+6+7+8+10+11',
		),

		# https://github.com/open-mmlab/mmpretrain/tree/main/configs/mocov3
		dict(
			name = 'VitBase-MoCo3-unsup',
			mmpretrain_id = 'mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k',
			combination = '+3+4+6+7',
		),
	])


	def __init__(self, cfg):
		self.config = cfg
		self.device = torch.device('cuda:0')

		net = mmpretrain.get_model(cfg.mmpretrain_id, pretrained=True, device=self.device)
		net = net_install_attn_export(net, verbose_layers=False)

		self.net = net





class FeatureCorrelationPassthrough(torch.nn.Module, AttnExportMixin):
	
	extract_multihead = False

	def __init__(self, replacement_shared):
		super().__init__()

		lid = replacement_shared['layer_idx']
		self.extra_output_storage = replacement_shared['extra_output_storage']
		self.module_path = f'feature_corr_{lid:02d}'
		replacement_shared['layer_idx'] += 1
		
		

	def forward(self, features):
		print(self.module_path, features.shape)

		corr = features @ features.transpose(-2, -1)
		corr *= 1e-3 / features.shape[2]

		self.export_attention(corr)

		return features


@MethodRegistry.register_class()
class FeatureEntropy_SingleImage_MMPretrain(AttEntropy_SingleImage):
	
	configs = extend_with_nocombination_configs([
		# ViT with imagenet pretraining
		# https://github.com/open-mmlab/mmpretrain/tree/main/configs/vision_transformer
		dict(
			name = 'FeatentVitBase-ImgNet',
			mmpretrain_id = 'vit-base-p16_in21k-pre_3rdparty_in1k-384px',
			combination = '+1+2+3+4+5+6+7+8+9+10',
		),
		dict(
			name = 'FeatentVitLarge-ImgNet',
			mmpretrain_id = 'vit-large-p16_in21k-pre_3rdparty_in1k-384px',
			combination = '+2+3+5+6+7+8+9+10+11+12+13+14+15+16+17+21+22+23',
		),

		# MAE masked self-supervised
		# https://github.com/open-mmlab/mmpretrain/tree/main/configs/mae
		dict(
			name = 'FeatentVitBase-MAE-unsup',
			mmpretrain_id = 'mae_vit-base-p16_8xb512-amp-coslr-1600e_in1k',
			combination = '+5+7+9+10',
		),
		dict(
			name = 'FeatentVitLarge-MAE-unsup',
			mmpretrain_id = 'mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k',
			combination = '+4+5+9+10',
		),

		dict(
			name = 'FeatentVitBase-MAE-ImageNetCls',
			mmpretrain_id = 'vit-base-p16_mae-1600e-pre_8xb128-coslr-100e_in1k',
			combination = '+2+3+5+6+7+8+10+11',
		),

		# https://github.com/open-mmlab/mmpretrain/tree/main/configs/mocov3
		dict(
			name = 'FeatentVitBase-MoCo3-unsup',
			mmpretrain_id = 'mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k',
			combination = '+3+4+6+7',
		),
	])


	def __init__(self, cfg):
		self.config = cfg
		self.device = torch.device('cuda:0')

		net = mmpretrain.get_model(cfg.mmpretrain_id, pretrained=True, device=self.device)

		net.extra_output_storage = {}
		replacement_shared = dict(
			extra_output_storage = net.extra_output_storage,
			layer_idx = 0,
		)

		layers = []
		for bg_layer in net.backbone.layers:
			layers += [
				bg_layer,
				FeatureCorrelationPassthrough(replacement_shared),
			]
			net.backbone.out_indices[0] += 1

		net.backbone.layers = torch.nn.ModuleList(layers)
		print('out indices', net.backbone.out_indices, 'num layers', net.backbone.layers.__len__())

		self.net = net

	
from mmseg.apis import MMSegInferencer

@MethodRegistry.register_class()
class AttEntropy_SingleImage_MMSeg(AttEntropy_SingleImage):
	
	configs = extend_with_nocombination_configs([
		# ViT with imagenet pretraining
		# https://github.com/open-mmlab/mmpretrain/tree/main/configs/vision_transformer
		dict(
			name = 'VitLarge-SETR-ctc',
			mmseg_id = 'setr_vit-l_pup_8xb1-80k_cityscapes-768x768',
			combination = '+1+2+3+4+5+6+7+8+9+10+11+21+22+23',
		),
		# combination = "+0+1+2+3+4+5",
		# +1+2+3+4+5+6+7+8+22+23
		# +0+1+2+3+4+5+6+7+8+9+10+11+12+13+23
		dict(
			name = 'VitLarge-SETR-ctc_CAll',
			mmseg_id = 'setr_vit-l_pup_8xb1-80k_cityscapes-768x768',
		),
		dict(
			name = 'VitLarge-SETR-ade20k',
			mmseg_id = 'setr_vit-l_pup_8xb2-160k_ade20k-512x512',
			combination = '+1+2+3+4+5+6+7+8+9+10+11+13+21',
		),
	])


	def __init__(self, cfg):
		self.config = cfg
		self.device = torch.device('cuda:0')

		net_minf = MMSegInferencer(model=cfg.mmseg_id, device=self.device)
		net = net_install_attn_export(net_minf.model.backbone, verbose_layers=False)
		self.net = net

# Attn19:  q (8161, 1, 1024) k (8161, 1, 1024) v (8161, 1, 1024) atn (1, 8161, 8161)
# Attn20:  q (8161, 1, 1024) k (8161, 1, 1024) v (8161, 1, 1024) atn (1, 8161, 8161)
# Attn21:  q (8161, 1, 1024) k (8161, 1, 1024) v (8161, 1, 1024) atn (1, 8161, 8161)
