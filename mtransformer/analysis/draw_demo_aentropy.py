
import torch
from mtransformer.visualization.show_image_edit import adapt_img_data, image_montage_same_shape
import numpy as np

CCol = (75, 200, 75)

def calc_ae_sum(aes):
	
	max_side = max(ae.shape[0] for ae in aes)
	
	for ae in aes:
		if ae.shape[0] != ae.shape[1]:
			raise ValueError(f'Attention entropy size is not square but {tuple(ae.shape)}')
	
	def resize(ae):
		if ae.shape[0] < max_side:
			return torch.nn.functional.interpolate(ae[None, None], (max_side, max_side))[0, 0]
		else:
			return ae
	
	
	ae_sum = resize(aes[0].clone())
	for ae in aes[1:]:
		ae_sum += resize(ae)

	return ae_sum

def ae_to_heatmap(ae, side=256):
	heatmap = adapt_img_data(-ae.cpu().numpy())
	mag = side // heatmap.shape[0]
	return np.repeat(np.repeat(heatmap, mag, axis=0), mag, axis=1)

def draw_aes(out, num_cols=6):
	ae_heatmaps = [ae_to_heatmap(ae) for ae in out.attn_entropy]
	captions = [n.replace('backbone.layers.', '').replace('.attn', '') for n in out.attn_layer_names]
																			
	demo_ae = image_montage_same_shape(ae_heatmaps, num_cols=num_cols, captions=captions, caption_size=1.5, border=8, caption_color=CCol)
	return demo_ae


def draw_demo_aentropy(out):
	ae_sum = calc_ae_sum(out.attn_entropy).cpu().numpy()

	col1 = image_montage_same_shape([
		out.seg_overlay[::2, ::2],
		np.repeat(np.repeat(-ae_sum, 2, axis=0), 2, axis=1),
		-out.seg_logsumexp[::2, ::2].cpu().numpy(),
	], captions = ['', '- AE sum', '- seg logsumexp'], border=16, num_cols=1, caption_color=CCol)

	demo = image_montage_same_shape([col1, draw_aes(out, num_cols=4)], captions=[out.get('fid', ''), ''], caption_color=CCol)
	return demo
