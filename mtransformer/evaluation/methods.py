

from road_anomaly_benchmark.datasets.dataset_registry import Registry

MethodRegistry = Registry()

def populate_registry():
	try:
		from ..mmseg_overrides.entropy_methods_mmpretrain import AttEntropy_SingleImage_MMPretrain
	except ImportError:
		print('Failed to load mmpretrain')

	try:
		from ..mmseg_overrides.entropy_methods_dino import AttEntropy_SingleImage_Dino
	except ImportError:
		print('Failed to load dino')

	try:
		from ..mmseg_overrides.setr_entropy import SETR_AttnEntropyOutput
	except ImportError:
		print('Failed to load SETR')

	try:
		from ..mmseg_overrides.segformer_attention import Segformer_Attentropy
	except ImportError:
		print('Failed to load Segformer')
