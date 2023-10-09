from pathlib import Path
from math import sqrt, ceil
from easydict import EasyDict
import torch
import numpy as np
from matplotlib import pyplot
import cv2 as cv

from ..visualization.show_image_edit import show, adapt_img_data, image_montage_same_shape, imread, imwrite

HIGHLIGHT_COLOR = (60, 150, 180)
HIGHLIGHT_WIDTH = 8


def mag_img(img, desired_side=256):
	mag = desired_side // img.shape[0]
	return np.repeat(np.repeat( img, mag, axis=0), mag, axis=1)

def mag_torch(img, desired_side=256):
	mag = desired_side // img.shape[0]
	return torch.repeat_interleave(torch.repeat_interleave( img, mag, dim=0), mag, dim=1)


def plot_layers(entropies, image, num_cols=4, joint_norm=True, comb_manual=None, comb_opt=None, layer_step=1, manual_label='manual'):
	# num_layers = net_out.attn_layers.__len__()
	entropies = list(entropies) # prevent overwriting array
	num_layers = entropies.__len__()
	max_side = max([e.shape[0] for e in entropies])

	ent_sums = {
		n: torch.zeros((max_side, max_side), dtype=torch.float32, device=entropies[0].device)
		for n in ['all', 'manual', 'opt']
	}

	for i, e in enumerate(entropies):
		e_resized = torch.nn.functional.interpolate(e[None, None], (max_side, max_side))[0, 0]
		
		ent_sums['all'] += e_resized
		
		if comb_manual is not None:
			ent_sums['manual'] += comb_manual[i] * e_resized
		
		if comb_opt is not None:
			ent_sums['opt'] += comb_opt[i] * e_resized
		
	captions_extra = ['', 'all']
	ent_sums['all'] *= (1./num_layers)
	entropies.append(ent_sums['all'])
	
	if comb_manual is not None:
		captions_extra.append(manual_label)
		ent_sums['manual'] *= float(1. / np.sum(comb_manual))
		entropies.append(ent_sums['manual'])
		
	if comb_opt is not None:
		captions_extra.append('opt')
		ent_sums['opt'] *= float(1. / np.sum(comb_opt))
		entropies.append(ent_sums['opt'])
	
	# 	ent_sum = entropies[0].clone()
	# 	for e in entropies[1:]:
	# 		ent_sum += e
	# 	ent_sum *= (1./num_layers)
	# 	entropies.append(ent_sum)

	# Joint normalization
	if joint_norm:
		# ent_all_long = torch.cat(entropies, dim=0)
		#ent_img_long = adapt_img_data(ent_all_long.cpu().numpy())

		ent_all_long = np.concatenate([mag_img(e.cpu().numpy(), max_side) for e in entropies], axis=0)
		ent_img_long = adapt_img_data(ent_all_long)
		ent_imgs = [ent_img_long[i*max_side:(i+1)*max_side] for i in range(entropies.__len__())]
	
		ent_imgs = [mag_img(ei) for ei in ent_imgs]
	else:
		ent_imgs = [mag_img(adapt_img_data(e.cpu().numpy())) for e in entropies]

	# ent_sum_img = ent_imgs.pofrom math import sqrt, ceil

		
	# ent_imgs.insert(0, cv.resize(fr.image, ent_imgs[1].shape[:2][::-1]))
	# captions.insert(0, '')
	# ent_imgs.insert(5, ent_sum_img)
	# captions.insert(5, 'average')
	# ent_imgs.insert(10, None)
	# captions.insert(10, '')
	
	
	if comb_manual is not None:
		# highlight_color = (30, 30, 130)
		highlight_color = HIGHLIGHT_COLOR
		w = HIGHLIGHT_WIDTH
		
		for c, ent_img, in zip(comb_manual, ent_imgs):
			if c != 0:
				ent_img[:w, :] = highlight_color
				ent_img[:, :w] = highlight_color
				ent_img[-w:, :] = highlight_color
				ent_img[:, -w:] = highlight_color
	
	montage_opts = dict(
		border=6, border_color=(255, 255, 255),
		caption_color=(50, 200, 50), caption_size=1, 
	)
	
	captions = [str(i) for i in range(0, num_layers, layer_step)]
	demo_img = image_montage_same_shape(ent_imgs[0:num_layers:layer_step], num_cols=num_cols, captions=captions, **montage_opts)
	# show([ent_imgs[0], ent_imgs[1]])
	
	
		
	
	num_layer_rows = ceil(captions.__len__() / num_cols)
	num_empty_rows = num_layer_rows - captions_extra.__len__()
	empty_rows = [None]*num_empty_rows
	demo_img2 = image_montage_same_shape(
		[cv.resize(image, ent_imgs[1].shape[:2][::-1])] + ent_imgs[num_layers:] + empty_rows, 
		num_cols=1,
		captions=captions_extra + empty_rows,
		**montage_opts,
	)
	
	h = demo_img.shape[0]
	gap = np.full((h, 32, 3), 255, dtype=np.uint8)
	demo_all = np.concatenate([demo_img2, gap, demo_img], axis=1)
	
	return demo_all





def entropy_stats_for_synth_obstacle(fr, factor=1.2, save_dir=None):
	
	mask_obj = fr.label_pixel_gt.astype(bool)
	mask_bg = np.logical_not(mask_obj)
	mask_dimension = mask_bg.shape
		
	entropies = [ent.cpu().numpy() for ent in fr.entropies]
	num_layers = entropies.__len__()
	
	avgs_obj = []
	avgs_bg = []
	
	for i, ent in enumerate(entropies):
		ent = cv.resize(ent, mask_dimension[::-1])
		
		# show([mask_obj, mask_bg, ent])
		
		nanamount = np.count_nonzero(np.isnan(ent))
		# print(i, 'nan', nanamount, 'totm', np.mean(ent))
		
		avg_obj = np.mean(ent[mask_obj])
		avg_bg = np.mean(ent[mask_bg])
		
		# print(i, avg_bg, avg_obj)
		
		avgs_obj.append(avg_obj)
		avgs_bg.append(avg_bg)
		
		
	avgs_obj, avgs_bg = np.array(avgs_obj), np.array(avgs_bg)
	
	selected_layers_mask = avgs_obj * factor < avgs_bg
	selected_layers = np.where(selected_layers_mask)[0]
	
	print(selected_layers)
	comb_str = ''.join(f'+{i}' for i in selected_layers)
	print(f'combination for factor {factor} is {comb_str}')
	

	
	plot_x = range(1, num_layers+1)
	# pyplot.scatter(plot_x, avgs_obj, label='object entropy')
	# pyplot.scatter(plot_x, avgs_bg, label='background entropy')
	# pyplot.legend()
	xs = range(num_layers)
	
	
	demo_layers = plot_layers(
		fr.entropies,
		image = fr.image,
		comb_manual = selected_layers_mask,
		manual_label='auto',
	)
	
	fig, plot = pyplot.subplots(1, 1, figsize=(8, 4))
	plot.plot(xs, avgs_obj, label='object entropy', marker='x', color='green')
	plot.plot(xs, avgs_bg, label='background entropy', marker='o', color='orange')
	plot.legend()
	
	plot.set_xlabel('layer number')
	plot.set_ylabel('average entropy')
	
	plot.set_xticks(xs)
	plot.grid(axis='x')
	fig.tight_layout()
	
	for i in selected_layers:
		pyplot.axvspan(i-0.5, i+0.5, facecolor=(HIGHLIGHT_COLOR[0]/255, HIGHLIGHT_COLOR[1]/255, HIGHLIGHT_COLOR[2]/255), alpha=0.2)
	
	if save_dir:
		savename = f'{fr.fid}_{fr.method_name}_f{factor}'
		
		dir_out = save_dir
		dir_out.mkdir(parents=True, exist_ok=True)
		
		for fmt in ['png', 'pdf']:
			pyplot.savefig(dir_out / f'{savename}__levels.{fmt}')

		for fmt in ['png', 'jpg']:
			imwrite(dir_out/f'{savename}__layers.{fmt}', demo_layers)
	
	return EasyDict(
		# layer combination str
		combination = comb_str,
		demo_img = demo_layers, 
		plot_fig = fig,
	)


# from kornia.utils import image_to_tensor
# from kornia.geometry import pyrdown



# def fr_for_analysis(fr):
#     img_tr = image_to_tensor(fr.image).float() * (1./255.)
#     # img = pyrdown(img[None])
#     img_tr = img_tr[None].to(device)
#     # img.shape, img.dtype

#     entropies = process_image_and_extract_attention(net, img_tr)

#     fr = EasyDict(fr)
#     fr.entropies = entropies
#     fr.method_name = 'metname'

#     return fr



def read_mask_optional(mask_path):
	mask_path = Path(mask_path)
	
	if mask_path.is_file():
		mask_img = imread(mask_path)
		# alpha threshold
		return (mask_img[:, :, 3] > 10).astype(np.uint8)

    # raise 
	return None


def get_testpattern_dset():
	dset_hole = [
		EasyDict(
			fid = f'TexTest{i}',
			image = imread(f'./texture_test/hole_v{i}.jpg'),
			label_pixel_gt = read_mask_optional(f'./texture_test/hole_v{i}_mask.png'),
		)
		for i in [5,4]
	]

	return dset_hole



from ..evaluation.methods import MethodRegistry
from ..evaluation.setr_entropy_segme import populate_registry
from ..paths import DIR_DATA
import click

DEFAULT_SAVE = DIR_DATA / '2021_AttnAutoLayers' 

def run_layer_choice(method_or_name, factor=1.1, save=DEFAULT_SAVE):
	
	dset_testpattern = get_testpattern_dset()
	fr = EasyDict(dset_testpattern[0])

	if isinstance(method_or_name, str):
		populate_registry()
		method = MethodRegistry.get(method_or_name)
	else:
		method = method_or_name

	fr.method_name = method.name
	fr.entropies = method.inference_layers(fr.image)

	print(f'Layer analysis for {method.name}')
	if save:
		print(f' Writing to {save}')
	out = entropy_stats_for_synth_obstacle(fr, factor, save_dir = save)

	# Check combination

	method_comb = method.config.get('combination')
	if method_comb is None:
		print(f'Method {method.name} has no layer combination stored')
	else:
		if method_comb == out.combination:
			print(f'Method {method.name}: combination {out.combination} matches config')
		else:
			print(f'Method {method.name}: combination calculated = {out.combination}, combination in config = {method_comb}')

	show(out.demo_img)


@click.command()
@click.argument('method_name', type=str)
@click.option('--factor', type=float, default=1.1)
@click.option('--save', type=click.Path(dir_okay=True, file_okay=False, exists=True), default=DEFAULT_SAVE)
def run_layer_choice_cli(method_name, factor=1.1, save=DEFAULT_SAVE):
	run_layer_choice(method_name, factor, save)
	
if __name__ == '__main__':
	run_layer_choice_cli()

