
from functools import lru_cache

import numpy as np
import torch
from kornia import morphology
from kornia.utils import image_to_tensor, tensor_to_image

@lru_cache(maxsize=32)
def get_morpho_kernel(ks=5, device=torch.device('cuda:0')):
	return torch.ones((ks, ks), dtype=torch.float32, device=device)


def semantic_overlay(image_np, seg_class, colors):
	dev = seg_class.device
	
	with torch.no_grad():
		if isinstance(colors, torch.Tensor) and colors.device == dev:
			...
		else:
			colors = torch.tensor(colors, dtype=torch.float32, device=dev)
			
		h, w = seg_class.shape
		img_cls_colorimg = torch.zeros((3, h, w), dtype=torch.float32, device=dev)
		img_transparency = torch.zeros((h, w), dtype=torch.float32, device=dev)
		img_scene = image_to_tensor(image_np).float().to(dev)

		for cl in range(colors.__len__()):
			mask = seg_class == cl
			mask_float = mask.float()

			cls_transparency = mask_float - 0.75*morphology.erosion(mask_float[None, None], get_morpho_kernel(11, dev))[0, 0]
			img_transparency += cls_transparency
			img_cls_colorimg += mask_float[None] * colors[cl][:, None, None]

		img_out = img_scene * (1.-img_transparency[None]) + img_cls_colorimg * img_transparency[None]
		img_out_np = tensor_to_image(img_out).astype(np.uint8)
		return img_out_np