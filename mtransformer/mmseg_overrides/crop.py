"""
Preparing dataset frames for separation eval

- crop image to a square and resize to the input shape of the network
	768^2 for SETR, 1024^2 for Segformer
- crop gt labels accordingly and resize them to the shape of attention map
	We will use 256^2 and resize attention maps to match (they range from 256^2 to 32^2)

"""

import numpy as np
import cv2 as cv
from easydict import EasyDict

def resize_gt(sem_label, attn_shape, obstacle_class_id = 1, road_class_id=0):
	"""
	"""
	
	obstacle_mask = sem_label == obstacle_class_id
	road_mask = (sem_label == road_class_id) | obstacle_mask
	
	_, instance_ids, instance_meta, instance_centers = cv.connectedComponentsWithStats(obstacle_mask.astype(np.uint8))
	
	num_objects = instance_meta.__len__() - 1
	
	instance_ids_resized = np.zeros(attn_shape, dtype=np.uint8)
	road_mask_resized = cv.resize(road_mask.astype(np.uint8), attn_shape, interpolation=cv.INTER_NEAREST)
	bboxes = np.zeros((num_objects + 1, 4), dtype=np.int32)
	
	for obj_id in range(1, num_objects+1):
		mask = instance_ids == obj_id
		mask_relative_area = cv.resize(mask.astype(np.uint8)*255, attn_shape, interpolation=cv.INTER_AREA)
		mask_thr = mask_relative_area > 255 // 6
		bboxes[obj_id] = cv.boundingRect(mask_thr.astype(np.uint8))
		
		instance_ids_resized[mask_thr] = obj_id
				
		# if b_show:		
		# 	print('Box: ', bboxes)
		# 	show([
		# 		cv2.resize(mask.astype(np.uint8), attn_shape, interpolation=cv2.INTER_NEAREST),
		# 		mask_relative_area, 
		# 		mask_thr,
		# 	])
			
		
	return EasyDict(
		gt_road_mask = road_mask_resized,
		gt_instance_ids = instance_ids_resized,
		gt_instance_box = bboxes, # [left, top, width, height]
	)
		

def frame_crop_resize_to_input(fr, input_shape, attn_shape = (256, 256)):
	"""
		
	"""
	
	in_h, in_w = fr.image.shape[:2]
	
	crop_h, crop_w = input_shape
	
	
	offset_x = (in_w - crop_w) // 2
	offset_y = (in_h - crop_h)
	
	sl = slice(offset_y, offset_y+crop_h), slice(offset_x, offset_x+crop_w)
	in_img = fr.image[sl]
	fr.image = in_img

	in_labels = fr.get('label_pixel_gt', None)
	if in_labels is not None:
		in_labels = in_labels[sl]
		fr.label_pixel_gt = in_labels
		fr.gt = resize_gt(in_labels, attn_shape)
	
	fr.fid = fr.fid.replace('/', '__')
	fr.fidnum = fr.get('fidnum', 1)
	
	return fr
	
	