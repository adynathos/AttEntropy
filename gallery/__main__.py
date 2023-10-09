
from pathlib import Path
import shutil
import click

IMG_SUFFIXES = {'.jpg', '.webp', '.png'}

TMPL_INVOCATION = """
	{GALLERY_SCRIPT}

	const IMAGE_DIR = "{IMAGE_DIR}"
	const IMAGE_FILES = [
{IMAGE_FILES}
	]
	run_gallery(IMAGE_DIR, IMAGE_FILES, [{IMG_W}, {IMG_H}])
"""


def web_gallery_generate(root_dir, image_dir, html_file, img_wh = (2048, 1024)):
	dir_root = Path(root_dir)
	dir_images = Path(image_dir)
	outfile = dir_root / html_file
	
	if not dir_images.is_absolute():
		dir_images = dir_root / dir_images

	img_names = [p.relative_to(dir_images) for p in  dir_images.rglob('*') if p.suffix.lower() in IMG_SUFFIXES]
	img_names.sort()
	
	CODE_DIR = Path(__file__).parent
	TMPL_HTML_FILE = (CODE_DIR / 'template.html').read_text()
	GALLERY_SCRIPT = (CODE_DIR / 'main.js').read_text()

	js_content = TMPL_INVOCATION.format(
		IMAGE_DIR = dir_images.relative_to(dir_root),
		IMAGE_FILES = ''.join([f'\t\t"{p}",\n' for p in img_names]),
		GALLERY_SCRIPT = GALLERY_SCRIPT,
		IMG_W = img_wh[0],
		IMG_H = img_wh[1],
	)

	html_content = TMPL_HTML_FILE.format(
		script_src = js_content,
	)

	
	outfile.write_text(html_content)
	try:
		shutil.copytree(CODE_DIR / 'lib', dir_root / 'lib', dirs_exist_ok=True)
	except TypeError:
		try:
			shutil.copytree(CODE_DIR / 'lib', dir_root / 'lib')
		except FileExistsError:
			...

	return outfile

@click.command()
@click.argument('root_dir', type=click.Path(dir_okay=True, file_okay=False, exists=True))
@click.argument('image_dir')
@click.option('--html-file', default='index.html')
def main(root_dir, image_dir, html_file):
	outfile = web_gallery_generate(root_dir, image_dir, html_file)	
	print(outfile)

if __name__ == '__main__':
	main()

"""

python -m gallery /cvlabdata2/home/lis/data/2004_AttnEntropySegMe/vis SETR-AttnEntropy_1+2+3+4+5+6+7+8+9+10+11+12+13+23 --html-file index_sum1.html


"""