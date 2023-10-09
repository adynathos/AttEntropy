import torch
import numpy as np
import cv2 as cv
from easydict import EasyDict

def calc_road_area_contour(sem_class_prediction, selected_classes, roi = None, **_):
	"""
	0 - non-road
	1 - road area (except for holes)
	2 - holes inside of road area
	255 - out of roi
	"""
	road_mask_with_holes = np.zeros_like(sem_class_prediction, dtype=bool)
	
	for c in selected_classes:
		road_mask_with_holes |= sem_class_prediction == c
	
	if roi is not None:
		road_mask_with_holes &= roi

	contours, _ = cv.findContours(
		image = road_mask_with_holes.astype(np.uint8),
		mode = cv.RETR_EXTERNAL,
		#method = cv.CHAIN_APPROX_TC89_L1,
		method = cv.CHAIN_APPROX_SIMPLE,
	)
	
	road_mask_filled = cv.drawContours(
		image = np.zeros_like(sem_class_prediction, dtype=np.uint8),
		contours = contours,
		contourIdx = -1, # all
		color = 1,
		thickness = -1, # fill
	)
	
	# show([road_mask_with_holes, road_mask_filled])
	
	road_label = np.zeros_like(sem_class_prediction, dtype=np.uint8)
	road_label[road_mask_filled.astype(bool)] = 2
	road_label[road_mask_with_holes] = 1 
	
	if roi is not None:
		road_label[np.logical_not(roi)] = 255
	
	return road_label


DILATE_KERNEL = np.ones((5, 5), dtype=np.uint8)
from functools import lru_cache
from math import sqrt

def attn_mask_to_road(seg_class, attn_maps):
	"""
	0 - non-road
	1 - road area (except for holes)
	2 - holes inside of road area
	255 - out of roi
	"""

	# seg_class = logits.argmax(dim=1)
	# labels: road = 0, sidewalk = 1

	assert seg_class.shape.__len__() == 2

	mask_road = seg_class <= 1

	mask_road = mask_road.cpu().byte().numpy()

	# masks = []

	# for b, mask_road_b in enumerate(mask_road):
	# mask_road_filled = np.zeros_like(mask_road_b, dtype=bool)

	contours, _ = cv.findContours(
		image = mask_road,
		mode = cv.RETR_EXTERNAL,
		#method = cv.CHAIN_APPROX_TC89_L1,
		method = cv.CHAIN_APPROX_SIMPLE,
	)

	road_mask_filled = cv.drawContours(
		image = np.zeros_like(mask_road, dtype=np.uint8),
		contours = contours,
		contourIdx = -1, # all
		color = 1,
		thickness = -1, # fill
	)

	mask_lowres_relatve_area = cv.resize(road_mask_filled*255, (192, 192), interpolation=cv.INTER_AREA)
	mask_lowres = mask_lowres_relatve_area > 255 // 4

	mask_lowres = cv.dilate(mask_lowres.astype(np.uint8), DILATE_KERNEL)

	# masks.append(mask_lowres)

	@lru_cache(maxsize=8)
	def get_mask_by_res(res):
		mask_attnres = cv.resize(mask_lowres*255, res, interpolation=cv.INTER_AREA)
		mask_attnres = mask_attnres > 255 // 4
		# print(res)
		# reshape to an attention row
		return torch.from_numpy(mask_attnres).reshape((1, -1)).float().to(attn_map[0].device)
	

	for attn_map in attn_maps:
		# attn_map = attn_map[b] # batchdim
		attn_map = attn_map[0] # batchdim
		tokens_squared, attn_squared = attn_map.shape

		attn_side = round(sqrt(attn_squared))

		attn_map *= get_mask_by_res((attn_side, attn_side))
		attn_map += 1e-8
		attn_map *= (1./attn_map.sum(dim=1, keepdims=True))

	return EasyDict(
		# road_masks = np.stack(masks, axis=0),
		road_mask = mask_lowres,
		attn_maps = attn_maps,
	)

