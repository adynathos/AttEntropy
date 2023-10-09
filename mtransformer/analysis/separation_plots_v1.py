
import torch
torch.set_grad_enabled(False)
# from sklearn.metrics import roc_auc_score

import torchmetrics

def analysis_vote_threshold(fr, failfids=None):
	MARGIN = 2
	THRESHOLDS = [0.0002, 0.001, 0.005, 0.01]
	N_patches = 48*48
	
	gt = fr.gt
	example_attn_map = fr.attention_maps.values().__iter__().__next__()
	dev = example_attn_map.device
	dev = 'cpu'
	gt_instance_id_torch = torch.from_numpy(gt.gt_instance_ids).to(dev).bool()
	gt_road_mask_torch = torch.from_numpy(gt.gt_road_mask).to(dev)
		
	# mask_box = torch.zeros_like(gt_road_mask_torch)
		
	num_obj = gt.gt_instance_box.__len__() - 1
	
	samples = []
	
	for layer_idx, (layer_name, attn) in enumerate(fr.attention_maps.items()):
		# drop 
		attn_nocls = attn[0, 1:, 1:]
			
		for thr in THRESHOLDS:
			attn_thr = (attn_nocls > thr).half()

			votes_incoming = attn_thr.sum(dim=1).reshape((48, 48)) * (1./N_patches)

			votes_incoming = votes_incoming.to(dev)
			
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

					# mask_box[:] = False
					# mask_box[box_slice] = True

					obj_mask_in_box = gt_instance_id_torch[box_slice]
					votes_in_box = votes_incoming[box_slice]  
				
					auroc = torchmetrics.functional.auroc(votes_in_box.reshape(-1), obj_mask_in_box.reshape(-1))
					# other_auroc = roc_auc_score(obj_mask_in_box.reshape(-1).numpy(), votes_in_box.reshape(-1).numpy())

					# print('auroc', auroc, [tuple(m.shape) for m in [mask_box, obj_mask_in_box, votes_in_box]])
					# show([m.cpu().numpy() for m in [mask_box, obj_mask_in_box, votes_in_box]])
					# show([fr.image, mask_box.cpu().numpy(), gt.gt_instance_ids])

					samples.append(dict(
						fidnum = fr.fidnum,
						obj_id = obj_id,
						attn_threshold = thr,
						layer = layer_idx,
						vote_auroc = float(auroc),
					))
				except Exception as e:
					# print(f'Frame {fr.fid} fail:', e)
					if failfids is not None:
						failfids.add((fr.fid, str(e)))
				
				
	return samples			
		
		

s1 = analyze_dset(dset_obs, analysis_vote_threshold, limit=24)


def analyze_votes(vote_stats_dicts):
	vote_sep_stats = pandas.DataFrame.from_dict(vote_stats_dicts)
	
	sep_stats_aggr = vote_sep_stats.groupby(['layer', 'attn_threshold'])['vote_auroc'].agg(['mean', 'var'])
	
	sep_stats_aggr['mean_optimal'] = np.maximum(sep_stats_aggr['mean'], 1-sep_stats_aggr['mean'])
	
	ser = sep_stats_aggr.unstack()

	for thr in [0.0002, 0.001, 0.005, 0.01]:
		# sep_stats_aggr.unstack()['mean_optimal'][thr].plot(kind='errorbar', label=f'thr {thr}')
		# seaborn.lineplot(y=ser['mean_optimal'][thr])
		ser['mean_optimal'][thr].plot(label=f'thr {thr}')

	pyplot.legend()
		
	# pyplot.figure()
	# seaborn.lineplot(y='vote_auroc', data=vote_sep_stats)
		
	
analyze_votes(s1)

# out = process_frame(dset_laf[5])
# out

def analysis_entropy_frame(fr, failfids=None):
	MARGIN = 2
	N_patches = 48*48
	
	gt = fr.gt
	example_attn_map = fr.attention_maps.values().__iter__().__next__()
	dev = example_attn_map.device
	dev = 'cpu'
	gt_instance_id_torch = torch.from_numpy(gt.gt_instance_ids).to(dev).bool()
	gt_road_mask_torch = torch.from_numpy(gt.gt_road_mask).to(dev)
	
	samples = []
	num_obj = gt.gt_instance_box.__len__() - 1

	for layer_idx, (layer_name, attn) in enumerate(fr.attention_maps.items()):

		attn_nocls = attn[0, 1:, 1:]
		attn_out_entropy = -(attn_nocls * torch.log(attn_nocls)).sum(dim=1)
		
		attn_out_entropy = attn_out_entropy.reshape(48, 48).to(dev)
			
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
					layer = layer_idx,
					scores_in_box = attn_out_entropy[box_slice],
					mask_in_box =  gt_instance_id_torch[box_slice],
				))
				
			except Exception as e:
				# print(f'Frame {fr.fid} fail:', e)
				if failfids is not None:
					failfids.add((fr.fid, str(e)))
					
	return samples			


def analysis_entropy_whole(framelist, limit=None, name=None):
	samples = analyze_dset(framelist, analysis_entropy_frame, limit=limit)
	
	name = name or framelist.name
	
	aurocs = []
	lids = list(range(24))
	
	for lid in lids:
		scores, masks = [
			torch.cat([s[key].reshape(-1) for s in samples if s['layer'] == lid], dim=0)
			for key in ['scores_in_box', 'mask_in_box']
		]
		
		auroc_inv = 1.- float(torchmetrics.functional.auroc(scores, masks))
		# auroc_inv = float(torchmetrics.functional.auroc(scores, masks))
		aurocs.append(auroc_inv)
	
	
	pyplot.xlabel('layer')
	pyplot.ylabel('1-AUROC')
	pyplot.title(f'Separation auroc for entropy {name}')
	pyplot.plot(lids, aurocs)
	
	
analysis_entropy_whole(dset_obs)