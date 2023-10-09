
fids_list = """
// Cityscapes
Cityscapes-val/frankfurt_000000_000294
Cityscapes-val/munster_000024_000019
Cityscapes-val/munster_000124_000019

// Obstacle - not segmented
ObstacleTrack-all/darkasphalt_watercanS_2
LostAndFound-trainValid/01_Hanns_Klemm_Str_45_000000_000260
LostAndFound-trainValid/12_Umberto_Nobile_Str_000003_000240

// Obstacle - segmented
ObstacleTrack-all/darkasphalt_canister_2
ObstacleTrack-all/darkasphalt_stump_1
LostAndFound-trainValid/12_Umberto_Nobile_Str_000001_000260

// Obstacle - non obstacle foreground texture
ObstacleTrack-all/curvy-street_carton_1
LostAndFound-trainValid/01_Hanns_Klemm_Str_45_000001_000150

// Obstacle - strong domain shift
ObstacleTrack-all/driveway_carton_bowl_3
ObstacleTrack-all/driveway_pig_vase_2
ObstacleTrack-all/one-way-street_backpack2_2
ObstacleTrack-all/snowstorm2_00_02_32.753

// Anomaly - mishmash
AnomalyTrack-all/airplane0000
AnomalyTrack-all/deer0000

// Anomaly - solid
AnomalyTrack-all/elephant0005	
AnomalyTrack-all/tractor0003
"""

# Cityscapes-val/frankfurt__frankfurt_000000_000294
# Cityscapes-val/munster__munster_000024_000019
# Cityscapes-val/munster__munster_000124_000019


# from mmseg_setr import infer_batch


def parse_fids_list(fl):
	fids_by_dset = {}
	
	for line in fl.split('\n'):
		line = line.strip()
		if line and not line.startswith('//'):
			dset, fid = line.split('/')
			fids_by_dset.setdefault(dset, []).append(fid)
			# fids.append((dset, fid))
	
	return fids_by_dset

EXEMPLARS = parse_fids_list(fids_list)

def iterate_over_exemplar_frames(func):
	from tqdm import tqdm
	from road_anomaly_benchmark.datasets import DatasetRegistry
	from .. import datasets # load extra datasets into registry

	results = []

	for dset_name, fids in EXEMPLARS.items():
		dset = DatasetRegistry.get(dset_name)
		
		for fid in tqdm(fids):			
			try:
				fr = dset[fid]
			except Exception as e:
				print(e)
				fr = dset.get_frame(fid, 'image')
			
			results.append(func(fr))

	return results
	