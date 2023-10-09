
from pathlib import Path
from functools import partial
from typing import Callable
import logging
from ..paths import DIR_MM_CONFIGS, DIR_MM_WEIGHTS
# from mmseg.apis import init_segmentor
from mmseg.apis import init_model


log = logging.getLogger(__name__)

# Base mmseg networks

def load_mmseg_from_cfg(net_def, device='cuda:0'):
	"""
	@param net_def: dict with keys `cfg` and `weight_file`
	"""
	if isinstance(net_def, str):
		net_def = DEFS_MMSEG_BASE[net_def]

	path_config = Path(net_def['cfg'] + '.py')
	path_config = path_config if path_config.is_absolute() else DIR_MM_CONFIGS / path_config

	path_weights = net_def['weight_file']
	if path_weights:
		path_weights = Path(path_weights)
		path_weights = path_weights if path_weights.is_absolute() else DIR_MM_WEIGHTS / path_weights

		log.info(f'Loading MMSeg network cfg={path_config} weights={path_weights}')

		return init_model(str(path_config), str(path_weights), device=device)
	else:
		log.info(f'Loading MMSeg network cfg={path_config} no weights')

		return init_model(str(path_config), device=device)


DEFS_MMSEG_BASE = {
	key: dict(cfg = key, weight_file = weights)
	for key, weights in {
	'pspnet_r50-d8_512x1024_40k_cityscapes': 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth',
	'setr_vit-large_pup_8x1_768x768_80k_cityscapes': 'setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth',
	'segformer_mit-b3_8x1_1024x1024_160k_cityscapes': 'segformer_mit-b3_8x1_1024x1024_160k_cityscapes_20211206_224823-a8f8a177.pth'
}.items()}

DEFS_MMSEG_BASE['vit_large_p16_384-b3be5167.pth'] = dict(
	cfg = 'setr_vit-large_pup_8x1_768x768_80k_cityscapes',
	weight_file = 'vit_large_p16_384-b3be5167.pth',
)

DEFS_MMSEG_BASE['dpt_imagenet'] = dict(
	cfg = 'dpt_vit-b16_512x512_160k_ade20k',
	weight_file = None,
)

DEFS_MMSEG_BASE['dpt_ade20k'] = dict(
	cfg = 'dpt_vit-b16_512x512_160k_ade20k',
	weight_file = 'dpt_vit-b16_512x512_160k_ade20k-db31cf52.pth',
)

DEFS_MMSEG_BASE['dpt_cityonly'] = dict(
	cfg = 'dpt_vit-b16_512x512_160k_cityscapes',
	weight_file = '/cvlabdata2/home/lis/exp_mmseg/dpt_uninit_ctc/latest.pth',
)

DEFS_MMSEG_BASE['setr_cityonly'] = dict(
	cfg = 'setr_vit-large_pup_8x1_768x768_80k_cityscapes',
	weight_file = '/cvlabdata2/home/lis/exp_mmseg/setr_uninit_ctc/latest.pth',
)

from .setr_attention import net_override_setr_attention

def load_setr_with_attention(net_def = DEFS_MMSEG_BASE['setr_vit-large_pup_8x1_768x768_80k_cityscapes'], device='cuda:0'):
	net = load_mmseg_from_cfg(net_def, device=device)
	net = net_override_setr_attention(net, verbose_patches=False)
	return net

# from .setr_entropy import SETR_AttnEntropyOutput

# def load_setr_entropy_eval(net_def = DEFS_MMSEG_BASE['setr_vit-large_pup_8x1_768x768_80k_cityscapes'], device='cuda:0'):
# 	net = load_mmseg_from_cfg(net_def, device=device)
# 	return SETR_AttnEntropyOutput.from_superclass(net)
