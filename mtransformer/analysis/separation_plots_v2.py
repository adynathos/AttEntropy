
from functools import partial
import torch, torchmetrics
tmt = torchmetrics.functional


def analysis_entropy_frame(fr, layer_combinations=None, failfids=None):
	MARGIN = 2
	N_patches = 48*48
	
	gt = fr.gt
	example_attn_map = fr.attention_maps.values().__iter__().__next__()
	dev = example_attn_map.device
	dev = 'cpu'
	gt_instance_mask_torch = torch.from_numpy(gt.gt_instance_ids).to(dev).bool()
	gt_road_mask_torch = torch.from_numpy(gt.gt_road_mask).to(dev).bool()
		
	def calc_entropy(attn):
		attn_nocls = attn[0, 1:, 1:]
		attn_out_entropy = -(attn_nocls * torch.log(attn_nocls)).sum(dim=1)
		attn_out_entropy = attn_out_entropy.reshape(48, 48).to(dev)
		return attn_out_entropy
	
	
	entropy_by_layer = [
		calc_entropy(attn)
		for layer_idx, (layer_name, attn) in enumerate(fr.attention_maps.items())
	]
	
	samples = []
	num_layers = entropy_by_layer.__len__()
	num_obj = gt.gt_instance_box.__len__() - 1

	# default is one per layer
	layer_combinations = layer_combinations or [str(i) for i in range(num_layers)]
	
	for comb_string in layer_combinations:
		layers = list(map(int, comb_string.split('+')))
		
		score = torch.zeros_like(entropy_by_layer[0])
		for lid in layers:
			score += entropy_by_layer[lid]
		
		samples.append(dict(
			fidnum = fr.fidnum,
			obj_id = -1,
			layer = comb_string,
			scores = score[gt_road_mask_torch],
			gt_mask =  gt_instance_mask_torch[gt_road_mask_torch],
			type = 'road',
		))

		
		for obj_id in range(1, num_obj+1):
			try:
				box = gt.gt_instance_box[obj_id]
				tl, wh = box[:2], box[2:4]
				# print(f'obj {obj_id}/{num_obj}', box)

				if wh[0]*wh[1] == 0:
					# object is smaller than a single patch
					continue			

				br = tl + wh

				# TODO what if obstacle hits image side?
				tl = tl - MARGIN
				br = br + MARGIN

				box_slice = slice(tl[1],br[1]), slice(tl[0],br[0])

				samples.append(dict(
					fidnum = fr.fidnum,
					obj_id = obj_id,
					layer = comb_string,
					scores = score[box_slice],
					gt_mask =  gt_instance_mask_torch[box_slice],
					type = 'box',
				))
				
			except Exception as e:
				# print(f'Frame {fr.fid} fail:', e)
				if failfids is not None:
					failfids.add((fr.fid, str(e)))
	
					
	return samples			


def analysis_entropy_whole(framelist, limit=None, dset_name=None):
	
	dset_name = dset_name or framelist.name
	
	lids = list(map(str, range(24))) + [
		'7+9',
		'8+10+11+13',
		'8+9+10+11+12+13',
	]

	
	samples = analyze_dset(
		framelist, 
		partial(analysis_entropy_frame, layer_combinations=lids),
		limit=limit,
	)
	
	metrics_names = EasyDict(
		auroc_box = 'AUROC in neighbourhood',
		prc_box = 'AP in neighbourhood',
		auroc_road = 'AUROC in road area',
		prc_road = 'AP in road area',
	)
	metrics = EasyDict({k: [] for k in metrics_names.keys()})
	
	for lid in lids:
		lid = str(lid)
		
		scores, masks = [
			torch.cat([
				s[key].reshape(-1) for s in samples 
				if s['layer'] == lid and s['type'] == 'road'
			], dim=0)
			for key in ['scores', 'gt_mask']
		]
		
		metrics.auroc_road.append(float(
			tmt.average_precision(-scores, masks)
		))
		
		metrics.prc_road.append(float(
			tmt.average_precision(-scores, masks)
		))

		
		scores, masks = [
			torch.cat([
				s[key].reshape(-1) for s in samples 
				if s['layer'] == lid and s['type'] == 'box'
			], dim=0)
			for key in ['scores', 'gt_mask']
		]
		
		metrics.auroc_box.append(float(
			tmt.auroc(-scores, masks)
		))
		metrics.prc_box.append(float(
			tmt.average_precision(-scores, masks)
		))

	
	num_plots = metrics_names.__len__()
	fig, plots = pyplot.subplots(1, num_plots, figsize=(16, 8))
	
	fig.suptitle(f'Separation for entropy on dataset {dset_name}')
	
	for plot, mk in zip(plots, metrics_names.keys()):
		name = metrics_names[mk]
		values = metrics[mk]
		
		plot.barh(lids, values)

		# plot.set_xlabel('layer')
		# plot.set_ylabel('1-AUROC')
		plot.invert_yaxis()	
		plot.set_title(name)
	
	
analysis_entropy_whole(dset_obs, limit=48)







def entropy_vis_frame(fr, layer_combinations=None, failfids=None, b_show=False):
	
	layer_combinations = [
		'1+2+3+4+5+6+7+8+9+10+11+12+13+23',
		'6+7+8+9+10+11+12+13',
		'7+9',
		'8+9+10+11',
		'8+10+11+13',
		'8+10+11+13+23',
		'8+9+10+11+12+13',
		'8+9+10+11+12+13+23',
		'+'.join(map(str, range(24))),
	]
	
	dev = 'cpu'
	MAG = 2
	COLS = 4
	
	def calc_entropy(attn):
		attn_nocls = attn[0, 1:, 1:]
		attn_out_entropy = -(attn_nocls * torch.log(attn_nocls)).sum(dim=1)
		attn_out_entropy = attn_out_entropy.reshape(48, 48).to(dev)
		return attn_out_entropy
	
	entropy_by_layer = [
		calc_entropy(attn)
		for layer_idx, (layer_name, attn) in enumerate(fr.attention_maps.items())
	]
	
	def vis_entropy(ent):
		ent = ent.cpu().numpy()
		
		return np.repeat(np.repeat(ent, MAG, axis=0), MAG, axis=1)
	
	# entropy of summed attn maps
	attn_maps_by_idx = list(fr.attention_maps.values())
			
	samples = []
	num_layers = entropy_by_layer.__len__()
	


	vis_single_layers = image_montage_same_shape(
		[vis_entropy(e) for e in entropy_by_layer],
		num_cols=COLS,
		border=4,
	)
	
	imgs_sums = []
	imgs_presum = []
	
	for comb_string in layer_combinations:
		layers = list(map(int, comb_string.split('+')))
		
		score = torch.zeros_like(entropy_by_layer[0])
		for lid in layers:
			score += entropy_by_layer[lid]
		
		imgs_sums.append(vis_entropy(score))
		
		
		attn_sum = torch.zeros_like(attn_maps_by_idx[0])
		for lid in layers:
			attn_sum += attn_maps_by_idx[lid]
		attn_sum *= (1./layers.__len__())
		
		imgs_presum.append(vis_entropy(calc_entropy(attn_sum)))
		del attn_sum
		
	
	vis_sums = image_montage_same_shape(
		imgs_sums,
		num_cols=COLS,
		border=4,
	)
	
	vis_presums = image_montage_same_shape(
		imgs_presum,
		num_cols=COLS,
		border=4,
	)
	
	vis_sums = image_montage_same_shape(
		[vis_sums, vis_presums],
		captions=['sums', 'presums'],
		caption_size = 1,
		num_cols=1,
	)
	
	demo = image_montage_same_shape(
		[fr.seg_overlay[::2, ::2], vis_single_layers, vis_sums],
		captions = ['', 'single', ''],
		caption_size = 1,
		num_cols=3,
		border=8,
	)
		
	if b_show:
		 show(demo)
	else:
		dir_imgs = DIR_DATA / '2003_AttnEntropy' / 'entropy_vis1'
		imwrite(dir_imgs / f'{fr.fid}_entropy.webp', demo)
		
def entropy_vis_dset(dset_name, fids):
	
	dset = DatasetRegistry.get(dset_name)
	
	
	for fid in tqdm(fids):
		fr = dset[fid]
		
		net_out = frame_prepare_for_analysis(fr)
		
	
		entropy_vis_frame(net_out)
		# return
	
for dset, fids in parse_fids_list(fids_list).items():
	print(dset, fids)
	# infer_batch(net_setr, dset, dir_base, 'pair1', vis_agg=False, vis_pair=True, vis_votes=True, fids=fids)
	entropy_vis_dset(dset, fids=fids)

	
	
