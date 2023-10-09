
from pathlib import Path
from os import environ
from operator import itemgetter
import logging

from easydict import EasyDict
from road_anomaly_benchmark.datasets.tracks import DatasetRegistry, DatasetRA, DIR_DATASETS, ChannelLoaderImage

log = logging.getLogger(__name__)

@DatasetRegistry.register_class()
class DatasetCityscapes(DatasetRA):
	
	DIR_CTC = Path(environ.get('DIR_CITYSCAPES', DIR_DATASETS / 'dataset_Cityscapes'))

	DEFAULTS = dict(
		dir_root=DIR_CTC,
		classes = dict(
			usual = (7, 8),
			anomaly = 100,
		)
	)

	configs = [
		dict(
			name = 'Cityscapes-train',
			split = 'train',
			expected_length = 2975,
			**DEFAULTS,
		),
		dict(
			name = 'Cityscapes-val',
			split = 'val',
			expected_length = 500,
			**DEFAULTS,
		),
	]

	channels = {
		'image': ChannelLoaderImage(
			'{dset.cfg.dir_root}/images/leftImg8bit/{dset.cfg.split}/{scene}/{fid}_leftImg8bit.{dset.img_fmt}',
		),
		'semantic_class_gt': ChannelLoaderImage(
			'{dset.cfg.dir_root}/gtFine/{dset.cfg.split}/{scene}/{fid}_gtFine_labelIds.png',
		),
		'instances': ChannelLoaderImage(
			'{dset.cfg.dir_root}/gtFine/{dset.cfg.split}/{scene}/{fid}_gtFine_instanceIds.png',
		),
	}

	LAF_SUFFIX_LEN = '_leftImg8bit'.__len__()

	@classmethod
	def frame_from_path(cls, path, **_):
		fid = path.stem[:-cls.LAF_SUFFIX_LEN] # removesuffix

		return EasyDict(
			fid = fid,
			scene = fid.split('_')[0],
		)


	def discover(self):
		img_dir = Path(self.cfg.dir_root) / 'images' / 'leftImg8bit' / self.cfg.split

		for img_ext in ['png', 'webp', 'jpg']:
			img_files = list(img_dir.glob(f'*/*_leftImg8bit.{img_ext}'))
			if img_files:
				break

		if not img_files:
			raise FileNotFoundError(f'{self.name}: Did not find images at {img_dir}')


		log.info(f'{self.name}: found images in {img_ext} format')
		self.img_fmt = img_ext

		# LAF's PNG images contain a gamma value which makes them washed out, ignore it
		# if img_ext == '.png':
			# self.channels['image'].opts['ignoregamma'] = True

		frames = [
			self.frame_from_path(p)
			for p in img_files
		]
		frames.sort(key = itemgetter('fid'))

		self.set_frames(frames)
		self.check_size()
