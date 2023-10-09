
from math import sqrt
import re
import torch

def determine_token_grid_shape(num_tokens, aspect_w_over_h = 1):
	"""
		h * w + 1 = num_tokens
		h * (aspect_w_over_h * h) = num_tokens - 1
		h^2 = (num_tokens - 1) / aspect_w_over_h
	"""

	# aspect_w_over_h = 640/480
	h = round(sqrt( num_tokens / aspect_w_over_h ))
	
	# w = round(aspect_w_over_h*h)
	w = num_tokens // h

	
	if w * h != num_tokens:
		raise AssertionError(f'Failure to reconstruct token grid for num_tokens={num_tokens} w/h={aspect_w_over_h}: {w}*{h} = {w*h}')

	return (h, w)


def attn_entropy(attn, aspect_w_over_h=1, b_remove_classtoken=True):

	if b_remove_classtoken:
		attn = attn[:, 1:, 1:]

	try:
		dim_B, num_tokens, num_attn = attn.shape
	except ValueError as e:
		e.args = e.args + (f'attn shape is {attn.shape}',)
		raise e

	# tokens_on_a_side = round(sqrt(num_tokens))

	token_grid_h_w = determine_token_grid_shape(num_tokens, aspect_w_over_h=aspect_w_over_h)

	out_sh = (dim_B, ) + token_grid_h_w

	# attention entropy
	attn_out_entropy = -(attn * torch.log(attn)).sum(dim=2) 
	attn_out_entropy = attn_out_entropy.reshape(out_sh)
	return attn_out_entropy


def process_image_and_extract_attentropy(net, image, b_remove_classtoken=True):

	# remove batch
	if image.shape.__len__() == 4:
		if image.shape[0] > 1:
			raise ValueError(f'Batched image')
		else:
			image = image[0]

	_, h, w = image.shape
	aspect_w_over_h = w/h

	with torch.no_grad():
		out = net(image[None])

		attn_layers = list(net.extra_output_storage.values())

		# if self.attn_road_masking_on:
		# 	seg_class = seg_logits.argmax(dim=1)
		# 	attn_layers = attn_mask_to_road(seg_class[0], attn_layers).attn_maps		

		# attn_ent_layers = [
		# 	- attn_entropy(attn, aspect_w_over_h=aspect_w_over_h) * weight
		# 	for (attn, weight) in 
		# 	zip(attn_layers, self.combination) if weight != 0
		# ]

		attn_ent_layers = [
			# drop batch dim
			attn_entropy(attn, aspect_w_over_h=aspect_w_over_h)[0]
			for attn in 
			attn_layers
		]

	return attn_ent_layers


def parse_combination_str(combination, num_layers=24):
	if combination == 'all':
		return [1] * num_layers
		
	layers = re.findall(r'([+-]\d+)', combination)
	combination = [0] * num_layers

	for layer in layers:
		sign = {'+': 1, '-': -1}[layer[0]]
		lid = int(layer[1:])
		combination[lid] = sign

	return combination


def process_image_and_extract_score(net, image, b_remove_classtoken=True, combination='all'):

	# remove batch
	if image.shape.__len__() == 4:
		if image.shape[0] > 1:
			raise ValueError(f'Batched image')
		else:
			image = image[0]


	_, h, w = image.shape
	aspect_w_over_h = w/h

	with torch.no_grad():
		out = net(image[None])

		attn_layers = list(net.extra_output_storage.values())

		combination_weights = parse_combination_str(combination, num_layers=len(attn_layers))

		# score = torch.zeros_like(attn_layers[0])
		score = None

		# efficiency: don't calc entropy on layers we don't use
		for attn, weight in zip(attn_layers, combination_weights):
		
			if weight != 0:
				ent = torch.squeeze(attn_entropy(attn, aspect_w_over_h=aspect_w_over_h))
				
				ent *= weight

				if score is None:
					score = ent
				else:
					score += ent

		# negate because low entropy detects objects
		score *= float( -1. / torch.sum( torch.abs( torch.tensor(combination_weights))) )

	return score



    
