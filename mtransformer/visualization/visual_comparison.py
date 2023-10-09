from pathlib import Path
import numpy as np
from easydict import EasyDict
import click
from .show_image_edit import imread, imwrite, image_montage_same_shape

from road_anomaly_benchmark.jupyter_show_image import get_boundary_mask

from road_anomaly_benchmark.datasets.dataset_registry import DatasetRegistry


def name_list(name_list):
	return [name for name in name_list.split(',') if name]


"""
/cvlabdata2/home/lis/data/2004_AttnEntropySegMe/vis_SegPred2/VitBase-Dino/ObstacleTrack-all/darkasphalt_basket_3.webp
"""


DIR_VIS = Path('/cvlabdata2/home/lis/data/2004_AttnEntropySegMe/vis_SegPred2')
DIR_OUT = Path('/cvlabdata2/home/lis/data/2004_AttnEntropySegMe/vis_comparison')


def load_vis_img(method, dset, fid):
	p = DIR_VIS / method / dset / f'{fid}.webp'
	return imread(p)

def cut_vis_img(vis_img):
	h, w, _ = vis_img.shape

	h_quarter = (h-4) // 2
	w_quarter = (w-4) // 2

	return EasyDict(
		image = vis_img[:h_quarter, :w_quarter],
		heatmap = vis_img[h_quarter+4:, :w_quarter],
	)

import cv2 as cv

@click.command()
@click.option('--name')
@click.option('--methods')
@click.option('--frames', help='format dset1.fid1,dset2.fid2')
def main(name, methods, frames):

	methods = name_list(methods)
	frames = name_list(frames)

	# populate dset registry
	from .. import datasets
	
	for dset_dot_fid in frames:
		print(dset_dot_fid)
		dset_name, fid = dset_dot_fid.split('.')

		dset = DatasetRegistry.get(dset_name)
		# dset.discover()

		# try:
		fr = dset[fid]
		# except KeyError as e:
		# 	print('key shit', e)
		# 	print(dset.frames_by_fid.__len__(), 'keys', list(dset.frames_by_fid.keys())[:5])

		photo = fr.image.copy()
		border = get_boundary_mask(fr.label_pixel_gt)
		border &= fr.label_pixel_gt != 255
		photo[border, 1] = 255
		
		grid = {
			fid: photo,
		}

		for m in methods:
			vis = cut_vis_img(load_vis_img(m, dset_name, fid))
			grid[m] = cv.pyrUp(vis.heatmap)

		demo_img = image_montage_same_shape(
			list(grid.values()),
			captions = list(grid.keys()),
			num_cols = 3,
			border = 4,
			caption_size = 1.5,
			downsample = 2,
		)

		imwrite(DIR_OUT / name / f'{name}_{fid}.webp', demo_img)


if __name__ == '__main__':
	main()

"""
export DIR_DATASETS="/cvlabsrc1/cvlab"
export DIR_OUTPUTS="/cvlabdata2/home/lis/data/2004_AttnEntropySegMe"


methods_vit_base=VitBase-Dino,VitBase-MAE-unsup,VitBase-ImgNet,DPT-AttnEntropy_manual
methods_vit_large=VitLarge-MAE-unsup,VitLarge-ImgNet,SETR-AttnEntropy_manual

frs=ObstacleTrack-all.darkasphalt_basket_3,LostAndFound-test.04_Maurener_Weg_8_000001_000170,MaSTr1325-low.0047,ArtificialLunarLandscape.0101

python -m mtransformer.visualization.visual_comparison --name VitBase --methods $methods_vit_base --frames $frs


python -m mtransformer.visualization.visual_comparison --name VitLarge --methods $methods_vit_large --frames $frs

python -m mtransformer.visualization.visual_comparison --name VitLarge --methods $methods_vit_large --frames ArtificialLunarLandscape.0011



"""
