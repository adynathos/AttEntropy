from pathlib import Path
from os import environ

DIR_THIS = Path(__file__).parent
DIR_REPO = DIR_THIS.absolute().parent
DIR_MM_WEIGHTS = Path(environ.get('DIR_MM_WEIGHTS', DIR_REPO.parent / 'checkpoints'))
DIR_MM_CONFIGS = Path(environ.get('DIR_MM_CONFIGS', DIR_THIS / 'mmseg_configs'))

DIR_DATA = Path(environ.get('DIR_DATA', DIR_REPO.parent / 'data'))
DIR_OUT = Path(environ.get('DIR_OUT', DIR_REPO.parent / 'out'))