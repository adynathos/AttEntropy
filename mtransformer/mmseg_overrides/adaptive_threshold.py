
import cv2 as cv
import numpy as np

def entropy_adaptive_thresholding(ent_sum_np):
	range_min, range_max = np.min(ent_sum_np), np.max(ent_sum_np)
	gap = range_max - range_min
	ent_sum_cv = ((ent_sum_np - range_min) * (255. / gap)).astype(np.uint8)
	
	
	# ent_sum_denoised = cv.medianBlur(ent_sum_cv, 3)
	KS = 3
	ent_sum_denoised = cv.GaussianBlur(ent_sum_cv, (KS, KS), 0)
	ent_sum_fused = ent_sum_cv.copy()
	ent_sum_fused[30:] = ent_sum_denoised[30:]
	
	return cv.adaptiveThreshold(
		ent_sum_fused, 
		1, 
		thresholdType=None, 
		adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
		blockSize = 41,
		C=-30,
	)
	