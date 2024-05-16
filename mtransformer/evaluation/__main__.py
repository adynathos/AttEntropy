

# Point SegmentMe bechmark code towards the dataset storage location
from pathlib import Path
from road_anomaly_benchmark.evaluation import Evaluation
from road_anomaly_benchmark.datasets.dataset_io import DatasetBase, ChannelLoaderImage
from road_anomaly_benchmark.datasets.dataset_registry import DatasetRegistry
from road_anomaly_benchmark.paths import DIR_OUTPUTS
from ..paths import DIR_OUT

# Interactive display
from tqdm import tqdm
import click
from matplotlib import cm
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .methods import MethodRegistry, populate_registry
from .island import calc_road_area_contour
from ..visualization.show_image_edit import imread, imwrite, adapt_img_data, image_montage_same_shape
from easydict import EasyDict


	

def segme_prepare_paths():
	tmp_output_storage = Path('/tmp/segme_anomaly_p')
	tmp_output_storage.mkdir(exist_ok=True)
	DIR_OUTPUTS.mkdir(exist_ok=True, parents=True)
	try:
		(DIR_OUTPUTS / 'anomaly_p').symlink_to(tmp_output_storage)
	except Exception as e:
		print(e)


CMAP = cm.get_cmap('plasma')

def visualize(fr, score, method_name='', save_dir=None):
	heatmap = adapt_img_data(score, cmap_pos=CMAP)

	mask_road = fr.label_pixel_gt <= 2
	
	fused_img = fr.image.copy()
	fused_img[mask_road] = heatmap[mask_road]
	
	demo_img = image_montage_same_shape(
		[fr.image, fused_img],
		captions = [fr.fid, method_name],
		border = 4,
		caption_size = 1.5,
		downsample = 2,
	)
	
	if save_dir:
		imwrite(save_dir / f'{fr.fid}.webp', demo_img)

	return demo_img


def visualize_ctc(fr, score, method_name='', save_dir=None, threshold =0.5):
	# heatmap = adapt_img_data(score, cmap_pos=CMAP)
	heatmap = adapt_img_data(-score)
	mask_road = fr.label_pixel_gt <= 2
	
	fused_img = fr.image.copy()
	fused_img[mask_road] = heatmap[mask_road]
	
	demo_img = image_montage_same_shape(
		[fr.image, heatmap, fused_img, score > threshold],
		captions = [fr.fid, method_name, '', f'thr = {threshold}'],
		border = 4,
		caption_size = 1.5,
		downsample = 2,
	)

	gt_img = np.full(fr.image.shape, 255, dtype=np.uint8)
	gt_img[mask_road] = (0, 0, 0)
	gt_img[fr.label_pixel_gt == 0] = (0, 0, 0)
	gt_img[fr.label_pixel_gt == 1] = (200, 0, 0)
	
	if save_dir:
		imwrite(save_dir / f'{fr.fid}_image.jpg', fr.image)
		imwrite(save_dir / f'{fr.fid}_heat.jpg', heatmap)
		imwrite(save_dir / f'{fr.fid}_gt.png', gt_img)
		imwrite(save_dir / f'{fr.fid}_all.webp', demo_img)

	return demo_img


MODE_1 = '1+2+3+4+5+6+7+8+9+10+11+12+13+23'

@click.group()
def main():
	...


class LocalImageDataset(DatasetBase):
	def discover(self, ):
		self.frames = [EasyDict(
			fid = Path(img_path).stem,
			img_path = img_path,
		) for img_path in image_paths]


	def __len__(self):
		return len(self.frames)
	
	def get_frame(self, idx_or_fid, *channels):
		if isinstance(idx_or_fid, int):
			fr = self.frames[idx_or_fid]
		else:
			fr = self.frames_by_fid[idx_or_fid]
		
		out_fr = EasyDict(fr, dset_name = self.cfg.get('name_for_persistence', self.cfg.name))

		channels = channels or self.channels.keys()

		for ch_name in channels:
			out_fr[ch_name] = self.channels[ch_name].read(dset=self, **fr)

		return out_fr



@main.command()
@click.option('--method', type=str, default=MODE_1)
@click.option('--threshold', type=float, default=0.5)
@click.option('--local-imgs', type=str, default="")
def heatmaps_local(method, local_imgs, threshold=0.5):
	segme_prepare_paths()		

	local_imgs = [Path(p) for p in local_imgs.split(',')]
	print('Checking inputs')
	for p in local_imgs:
		if not p.is_file():
			raise FileNotFoundError(p)


	populate_registry()
	net = MethodRegistry.get(method)
	method_name = method

	def infer(img):
		# return net.inference_custom(img).cpu().numpy()
		res =  net.inference_custom(img)
		# print(res.keys())
		return res.anomaly_p.cpu().numpy()
		# return net.inference_custom(img).cpu().numpy()


	with ThreadPoolExecutor(4) as pool:
		for img_path in tqdm(local_imgs):
			image = imread(img_path)
			result = infer(fr.image)

			fr = EasyDict(
				fid = img_path.stem,
				image = image,
			)

			# pool.submit(visualize_ctc, fr, result, method_name=method_name, threshold=threshold, save_dir=dir_vis)
			visualize_ctc(fr, result, method_name=method_name, threshold=threshold, save_dir=img_path.parent)


@main.command()
@click.option('--method', type=str, default=MODE_1)
@click.option('--dsets', type=str)
@click.option('--threshold', type=float, default=0.5)
@click.option('--fids', type=str)
@click.option('--local-imgs', type=str, default="")
def heatmaps(method, dsets, threshold=0.5, fids=None, local_imgs=""):
	segme_prepare_paths()

	print('Checking datasets')
	for dset_name in dsets.split(','):
		print('. ', dset_name)
		ds = DatasetRegistry.get(dset_name)
		ds[0]


	# net_setr = load_net()
	# net_setr.set_entropy_layers(method)
	# net_setr.set_variant(method)
	# method_name = net_setr.name

	populate_registry()
	net = MethodRegistry.get(method)
	method_name = method

	def infer(img):
		# return net.inference_custom(img).cpu().numpy()
		res =  net.inference_custom(img)
		# print(res.keys())
		return res.anomaly_p.cpu().numpy()
		# return net.inference_custom(img).cpu().numpy()


	


	for dset_name in dsets.split(','):
		# load dset separately to see road masks

		print(f"""
===
	{method_name} vs {dset_name}
===
		""")

		dset = DatasetRegistry.get(dset_name)
		dir_vis = DIR_OUTPUTS / 'vis' / method_name / dset_name

		if fids is None:
			fids_infer = [f.fid for f in dset.frames]
		else:
			fids_infer = fids.split(',')

		with ThreadPoolExecutor(4) as pool:
			

			for fid in tqdm(fids_infer):
				fr = dset[fid]
				# run method here
				result = infer(fr.image)

				# visualize
				# fr = dset[frame.fid]
				fr.fid = fr.fid.replace('/', '__')

				# pool.submit(visualize_ctc, fr, result, method_name=method_name, threshold=threshold, save_dir=dir_vis)
				visualize_ctc(fr, result, method_name=method_name, threshold=threshold, save_dir=dir_vis)


import h5py
def save_attn(path, attnvals):
	with h5py.File(path, 'w') as file_out:
		file_out['attention_layers'] = attnvals

	
from kornia.geometry.transform import pyrdown
def prepare_layers_for_saving(attnvals):
	resized = pyrdown(attnvals[None], factor=4)[0]
	return resized.half().cpu().numpy()

@main.command()
@click.option('--method', type=str)
@click.option('--dset', type=str)
@click.option('--dir_out', type=click.Path(), default=DIR_OUT)
def export_attn(method, dset, dir_out):
	dir_out = Path(dir_out)

	print('Checking datasets')
	print('. ', dset)
	try:
		ds = DatasetRegistry.get(dset)
		ds[0]
	except ValueError:
		print('Loading from directory', dset)
		dir_src = Path(dset)
		if not dir_src.is_dir():
			raise FileNotFoundError(f'{dir_src} is neither directory not dataset')
		
		paths = list(dir_src.glob('*.webp')) + list(dir_src.glob('*.jpg')) + list(dir_src.glob('*.png'))
		paths.sort()
		if paths:
			print(f'Found {paths.__len__()} frames at {dir_src}')
		else:
			raise FileNotFoundError(f'No images at {dir_src}')

		def iter_inputs():
			for p in paths:
				yield EasyDict(
					fid = p.stem,
					image = imread(p),
				)

		ds = iter_inputs()
		dset = ''


	populate_registry()
	net = MethodRegistry.get(method)

	dir_out = dir_out / dset / method
	dir_out.mkdir(parents=True, exist_ok=True)

	with ThreadPoolExecutor(8) as pool:
		for frame in tqdm(ds):
			net_out = net.inference_custom(frame.image)

			pool.submit(
				save_attn, 
				dir_out / f'{frame.fid}_ent.hdf5',
				prepare_layers_for_saving(net_out.entropy_layers),
			)

"""

python -m mtransformer.evaluation.setr_entropy_segme export-attn --method SETR-AttnEntropy_export --dset ObstacleTrack-all 
python -m mtransformer.evaluation.setr_entropy_segme export-attn --method SETR-AttnEntropy_export --dset LostAndFound-test

"""


@main.command()
@click.option('--method', type=str, default=MODE_1)
@click.option('--dsets', type=str, default='')
@click.option('--dsets-eval', type=str, default='')
@click.option('--island', type=float)
@click.option('--vis/--no-vis', default=True)
@click.option('--profile/--no-profile', default=False)
def full_eval(method, dsets, dsets_eval=None, island=None, vis=True, profile=False):
	"""
	Infer, save, run metrics, visualize.
	"""
	from .. import datasets
	segme_prepare_paths()
	
	dsets_eval = dsets_eval or dsets

	print('Checking datasets')
	for dset_name in set(dsets.split(',') + dsets_eval.split(',')).difference(['']):
		print('. ', dset_name)
		DatasetRegistry.get(dset_name)

	method_name = method
	# island = island or method_name.endswith('_Island')
	if method_name.endswith('_Island'):
		if island is None:
			island = 1.
		method_name = method_name.replace('_Island', '')

	if method_name.endswith('_Res768'):
		resize = 768
		method_name = method_name.replace('_Res768', '')
	elif method_name.endswith('_Res1024'):
		resize = 1024
		method_name = method_name.replace('_Res1024', '')
	else:
		resize = None

	# net_setr = load_net()
	# net_setr.set_entropy_layers(method)
	# net_setr.set_variant(method_name)

	populate_registry()
	net = MethodRegistry.get(method_name)

	if profile:
		net.profiler_enable(True)

	if island is not None:
		method_name += '_Island'
	
	if resize:
		import torch
		import cv2 as cv
		method_name += f'_Res{resize}'

	def infer(img):

		if resize is not None:
			img_shape_orig = img.shape[:2]
			img = cv.resize(img, (resize, resize))

		out = net.inference_custom(img)
		anomaly_p = out.anomaly_p
		
		if resize is not None:
			anomaly_p = torch.nn.functional.interpolate(anomaly_p[None, None], img_shape_orig)[0, 0]
		
		anomaly_p = anomaly_p.cpu().numpy()

		if island is not None:
			island_mask = calc_road_area_contour(out.seg_class.cpu().numpy(), (0, 1))
			anomaly_p[island_mask == 2] = island

		return anomaly_p

	for dset_name in dsets.split(','):
		print(dset_name, 'out of', dsets)
		if not dset_name:
			continue
		# load dset separately to see road masks

		print(f"""
===
	{method_name} vs {dset_name}
===
		""")

		dset = DatasetRegistry.get(dset_name)
		dir_vis = DIR_OUTPUTS / 'vis' / method_name

		ev = Evaluation(
			method_name = method_name, 
			dataset_name = dset_name,
		)

		# with ThreadPoolExecutor(4) as pool:
		for frame in tqdm(ev.get_frames()):
			# run method here
			result = infer(frame.image)
			# provide the output for saving
			ev.save_output(frame, result)

			# visualize
			fr = dset[frame.fid]
			fr.fid = fr.fid.replace('/', '__')

				# pool.submit(visualize, fr, result, method_name=method_name, save_dir=dir_vis)
			# visualize(fr, result, method_name=method_name, save_dir=dir_vis)

		if profile:
			net.profiler_print()


		# wait for the background threads which are saving
		ev.wait_to_finish_saving()


	for dsev in dsets_eval.split(','):
		if not dsev:
			continue

		print(f""" Metrics {dsev} """)
		ev = Evaluation(
			method_name = method_name, 
			dataset_name = dsev,
		)
		ag = ev.calculate_metric_from_saved_outputs(
			'PixBinaryClass',
		)
		ag = ev.calculate_metric_from_saved_outputs(
			'SegEval-ObstacleTrack',
			frame_vis = vis,
		)

		# for metric in ['PixBinaryClass', 'SegEval-ObstacleTrack']:
		# 	ag = ev.calculate_metric_from_saved_outputs(
		# 		metric,
		# 	)

	if vis:
		from gallery import web_gallery_generate
		web_gallery_generate(DIR_OUTPUTS / 'vis_SegPred', method_name, f'index_{method_name}.html')


if __name__ == '__main__':
	from .. import datasets
	main()

"""

METHOD="SETR-AttnEntropy_1+2+3+4+5+6+7+8+9+10+11+12+13+23"
METHOD="SETR-AttnEntropyDiffused_1+2+3+4+5+6+7+8+9+10+11+12+13+23"

METHOD="SETR-AttnEntropy_8+9+10+11"

python -m mtransformer.evaluation.setr_entropy_segme --method $METHOD --dsets ObstacleTrack-validation --dsets-infer ObstacleTrack-validation

python -m mtransformer.evaluation.setr_entropy_segme --method $METHOD --dsets ObstacleTrack-all --dsets-infer ObstacleTrack-test,ObstacleTrack-all,ObstacleTrack-validation,ObstacleTrack-night,ObstacleTrack-snowstorm &

python -m mtransformer.evaluation.setr_entropy_segme --method $METHOD --dsets LostAndFound-testNoKnown &
wait


METHOD="SETR-AttnEntropy_1+2+3+4+5+6+7+8+9+10+11+12+13+23"

python -m mtransformer.evaluation.setr_entropy_segme heatmaps --method $METHOD --dsets Cityscapes-train --threshold -87




export DIR_DATASETS="/cvlabsrc1/cvlab"
export DIR_OUTPUTS="/cvlabdata2/home/lis/data/2004_AttnEntropySegMe"

METHOD="SETR-AttnEntropy_manual"
python -m mtransformer.evaluation.setr_entropy_segme heatmaps --method $METHOD --dsets ArtificialLunarLandscape --fids 9321,8641,8581


METHOD="SETR-AttnEntropy_manual"
python -m mtransformer.evaluation.setr_entropy_segme heatmaps --method $METHOD --dsets MaSTr1325-low --fids 1032


METHOD="SETR-AttnEntropy_manual"
python -m mtransformer.evaluation.setr_entropy_segme heatmaps --method $METHOD --dsets ObstacleTrack-all --fids curvy-street_carton_6


METHOD="SETR-AttnEntropy_manual"
python -m mtransformer.evaluation.setr_entropy_segme heatmaps --method $METHOD --dsets ObstacleTrack-all --fids 12210ad7-83f8-4b54-bb4b-e93f8ff6ac1f


METHOD="SETR-AttnEntropy_all"
python -m mtransformer.evaluation.setr_entropy_segme heatmaps --method $METHOD --dsets ObstacleTrack-all --fids validation_14
METHOD="Segformer-AttnEnt_sumAll"
python -m mtransformer.evaluation.setr_entropy_segme heatmaps --method $METHOD --dsets ObstacleTrack-all --fids validation_14





"""