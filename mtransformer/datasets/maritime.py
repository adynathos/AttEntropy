
from pathlib import Path
from os import environ
from operator import itemgetter
import logging
import cv2 as cv

from easydict import EasyDict
from road_anomaly_benchmark.datasets.tracks import DatasetRegistry, DatasetRA, DIR_DATASETS, ChannelLoaderImage

log = logging.getLogger(__name__)

def resize_if_key_present(fr, key, width, mode=cv.INTER_LINEAR):
	val = fr.get(key)
	if val is not None:
		h, w = val.shape[:2]
		sz_y = round(width * h / w)
		fr[key] = cv.resize(val, (width, sz_y), interpolation=mode)
	
	return fr

@DatasetRegistry.register_class()
class DatasetMaSTR(DatasetRA):
	
	DIR_MASTR = Path(environ.get('DIR_MASTR', DIR_DATASETS / 'dataset_MaSTr1325'))

	DEFAULTS = dict(
		dir_root=DIR_MASTR,
		classes = dict(
			usual = (1,2),
			anomaly = 0,
		)
	)

	configs = [
		dict(
			name = 'MaSTr1325-768',
			# expected_length =,
			resize_width = 768,
			**DEFAULTS,
		),
		dict(
			name = 'MaSTr1325-low',
			# expected_length =,
			**DEFAULTS,
		),
	]

	channels = {
		'image': ChannelLoaderImage(
			'{dset.cfg.dir_root}/images/{fid}.jpg',
		),
		'semantic_class_gt': ChannelLoaderImage(
			'{dset.cfg.dir_root}/labels/{fid}m.png',
		),
	}

	def get_frame(self, key, *channels):

		fr = super().get_frame(key, *channels)

		res_width = self.cfg.get('resize_width')
		if res_width is not None:
			resize_if_key_present(fr, 'image', res_width, cv.INTER_LINEAR)
			resize_if_key_present(fr, 'label_pixel_gt', res_width, cv.INTER_NEAREST)
			resize_if_key_present(fr, 'semantic_class_gt', res_width, cv.INTER_NEAREST)

		return fr

	@classmethod
	def frame_from_path(cls, path, **_):
		fid = path.stem[:-cls.LAF_SUFFIX_LEN] # removesuffix

		return EasyDict(
			fid = fid,
			scene = fid.split('_')[0],
		)


	def discover(self):
		img_dir = Path(self.cfg.dir_root) / 'images'

		img_files_all = list(img_dir.glob(f'*.jpg'))

		if not img_files_all:
			raise FileNotFoundError(f'{self.name}: Did not find images at {img_dir}')

		img_files = [p for p in img_files_all if 'old' not in p.stem]

		log.info(f'{self.name}: found {img_files.__len__()} images, discarding {img_files_all.__len__() - img_files.__len__()} old ones')

		frames = [
			EasyDict(fid = p.stem)
			for p in img_files
		]
		frames.sort(key = itemgetter('fid'))

		self.set_frames(frames)
		# self.check_size()
