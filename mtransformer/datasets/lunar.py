
from pathlib import Path
from os import environ
from operator import itemgetter
import logging

import numpy as np
from easydict import EasyDict
from road_anomaly_benchmark.datasets.tracks import DatasetRegistry, DatasetRA, DIR_DATASETS, ChannelLoaderImage

log = logging.getLogger(__name__)

@DatasetRegistry.register_class()
class DatasetLunarLandscape(DatasetRA):
	"""
	https://www.kaggle.com/code/romainpessia/understanding-and-using-the-dataset-wip
	"""
	
	DIR_LL = Path(environ.get('DIR_LUNAR_LANDSCAPE', DIR_DATASETS / 'dataset_ArtificialLunarLandscape'))

	DEFAULTS = dict(
		dir_root=DIR_LL,
		name_for_persistence = 'ArtificialLunarLandscape',
		exclude_fids = [821, 1731, 1831, 4021, 5171]
	)

	configs = [
		dict(
			name = 'ArtificialLunarLandscape',
			# expected_length =,
			**DEFAULTS,
		),
		dict(
			name = 'ArtificialLunarLandscape_one10',
			frame_skip_step = 10,
			**DEFAULTS,
		),
	]

	channels = {
		'image': ChannelLoaderImage(
			'{dset.cfg.dir_root}/images/render/render{fid}.png',
		),
		'semantic_color_gt': ChannelLoaderImage(
			# '{dset.cfg.dir_root}/images/clean/clean{fid}.png',
			'{dset.cfg.dir_root}/images/ground/ground{fid}.png',
		),
	}


	def get_frame(self, idx_or_fid, *channels):
		channels = set(channels)
		wants_labels_explicitly = False
		if 'label_pixel_gt' in channels:
			wants_labels_explicitly = True
			channels.remove('label_pixel_gt')
			channels.add('semantic_class_gt')


		if isinstance(idx_or_fid, int):
			base_fr = self.frames[idx_or_fid]
		else:
			base_fr = self.frames_by_fid[idx_or_fid]
		
		fr = EasyDict(base_fr, dset_name = self.cfg.get('name_for_persistence', self.cfg.name))

		channels = channels or self.channels.keys()

		for ch_name in channels:
			fr[ch_name] = self.channels[ch_name].read(dset=self, **fr)


		sem_color_gt = fr.get('semantic_color_gt')
		if sem_color_gt is not None:
			# print(idx_or_fid, sem_color_gt.shape)
			h, w = sem_color_gt.shape[:2]
			
			# initialize to road
			label = np.full((h, w), 0, dtype=np.uint8)

			# obstacles are blue (big) and green (small)
			label[(sem_color_gt[:, :, 2] > 0) | (sem_color_gt[:, :, 1] > 0)] = 1
			# sky is red => out of ROI
			label[sem_color_gt[:, :, 0] > 0] = 255

			fr['label_pixel_gt'] = label

		elif wants_labels_explicitly:
			raise KeyError(f'No labels for {idx_or_fid} in {self}')


		return fr

	@classmethod
	def frame_from_path(cls, path, **_):
		fid = path.stem[:-cls.LAF_SUFFIX_LEN] # removesuffix

		return EasyDict(
			fid = fid,
			scene = fid.split('_')[0],
		)


	def discover(self):
		excludes = set(self.cfg.exclude_fids)

		frames = [
			EasyDict(fid = f'{fid:04d}')
			for fid in range(1, 9766+1, self.cfg.get('frame_skip_step', 1)) if fid not in excludes
		]

		self.set_frames(frames)
		self.get_frame(0)

