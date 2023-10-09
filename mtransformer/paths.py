from pathlib import Path
from os import environ

DIR_MM_WEIGHTS = Path(environ.get('DIR_MM_WEIGHTS', '/cvlabsrc1/cvlab/pytorch_model_zoo/mmlab'))
DIR_MM_CONFIGS = Path(environ.get('DIR_MM_CONFIGS', Path(__file__).parent / 'mmseg_configs'))

DIR_DATA = Path(environ.get('DIR_DATA', '/cvlabdata2/home/lis/data'))
