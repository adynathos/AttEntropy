

from road_anomaly_benchmark.datasets.dataset_registry import Registry

MethodRegistry = Registry()

def populate_registry():
	try:
		from ..mmseg_overrides.entropy_methods_mmpretrain import AttEntropy_SingleImage_MMPretrain
	except ImportError as e:
		print('Failed to load mmpretrain', e)

	try:
		from ..mmseg_overrides.entropy_methods_dino import AttEntropy_SingleImage_Dino
	except ImportError as e:
		print('Failed to load dino', e)

	try:
		from ..mmseg_overrides.setr_entropy import SETR_AttnEntropyOutput
	except ImportError as e:
		print('Failed to load SETR', e)

	try:
		from ..mmseg_overrides.segformer_attention import Segformer_Attentropy
	except ImportError as e:
		print('Failed to load Segformer', e)
