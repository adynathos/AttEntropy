
from easydict import EasyDict
import numpy as np
import torch
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
# from mmseg.apis import inference_segmentor
from mmseg.apis import inference_model


from . import obj_copy_attrs
from ..visualization.show_image_edit import show
from ..visualization.semantic_overlay import semantic_overlay

def softmax_entropy(logits):
	return - torch.sum(logits * torch.log(logits + 1e-12), dim=0)

# log sum exp
def logits_to_logsumexp(logits):
	logits_exp = torch.exp(logits)
	sum_over_class = torch.sum(logits_exp, dim=0) # no batch
	return torch.log(sum_over_class + 1e-12)


def infer_image(net, image_np, b_numpy=True, b_show=True):
	out = EasyDict()

	# this requires the network object to be overriden to output logits
	out.seg_logits = inference_model(net, image_np)[0]

	with torch.no_grad():
		out.seg_softmax = torch.nn.functional.softmax(out.seg_logits, dim=0) # no batch dim
		out.seg_entropy = softmax_entropy(out.seg_softmax)
		out.seg_class = torch.argmax(out.seg_softmax, dim=0)
		out.seg_logsumexp = logits_to_logsumexp(out.seg_logits)
		out.seg_overlay = semantic_overlay(image_np, out.seg_class, net.PALETTE)
		
		logsum_from_zero = out.seg_logsumexp - torch.min(out.seg_logsumexp)
	
		out.logsum_inverted = logsum_from_zero / torch.max(logsum_from_zero)
		out.logsum_inverted = 1. - out.logsum_inverted

	if b_numpy:
		for k, v in out.items():
			if isinstance(v, torch.Tensor):
				out[k] = v.cpu().numpy()
		
	if b_show:
		show(
			[image_np, out.seg_overlay], 
			[out.seg_entropy, out.logsum_inverted],
		)
		
	return out


class EncoderDecoder_LogitOutput(EncoderDecoder):
	"""
	Modify MMseg's network class to output un-softmaxed logits.
	"""

	def __init__(self):
		"""
		The attributes are copied from a network object loaded by MM,
		and not initialized in this constructor.
		"""
		...
	
	def inference(self, img, img_meta, rescale):
		""" override to not apply softmax """

		assert self.test_cfg.mode in ['slide', 'whole']
		ori_shape = img_meta[0]['ori_shape']
		assert all(_['ori_shape'] == ori_shape for _ in img_meta)
		if self.test_cfg.mode == 'slide':
			seg_logit = self.slide_inference(img, img_meta, rescale)
		else:
			seg_logit = self.whole_inference(img, img_meta, rescale)
		
		# output = F.softmax(seg_logit, dim=1)
		output = seg_logit
		
		flip = img_meta[0]['flip']
		if flip:
			flip_direction = img_meta[0]['flip_direction']
			assert flip_direction in ['horizontal', 'vertical']
			if flip_direction == 'horizontal':
				output = output.flip(dims=(3, ))
			elif flip_direction == 'vertical':
				output = output.flip(dims=(2, ))

		return output

	def simple_test(self, img, img_meta, rescale=True):
		""" override to not cast to numpy nor list """
		seg_logit = self.inference(img, img_meta, rescale)
		# seg_pred = seg_logit.argmax(dim=1)
		seg_pred = seg_logit

		# if torch.onnx.is_in_onnx_export():
		# 	# our inference backend only support 4D output
		# 	seg_pred = seg_pred.unsqueeze(0)
		# 	return seg_pred
		# seg_pred = seg_pred.cpu().numpy() # removed
		# unravel batch dim
		# seg_pred = list(seg_pred) # removed
		return seg_pred

	@classmethod
	def from_superclass(cls, orig_net):
		my_net = cls()
		obj_copy_attrs(my_net, orig_net)
		return my_net

	def inference_custom(self, image):
		return infer_image(self, image, b_show=False)

def net_override_logits_mode(net_orig):
	net_modified = EncoderDecoder_LogitOutput.from_superclass(net_orig)
	return net_modified

