
import PhotoSwipeLightbox from './lib/photoswipe-lightbox.esm.min.js'


function run_gallery(IMAGE_DIR, IMAGE_FILES, width_height) {
	if (width_height === undefined) {
		width_height = [2048, 1024]
	}

	const [width, height] = width_height

	const options = {
		dataSource: IMAGE_FILES.map((name) => { return {
		 	src: `${IMAGE_DIR}/${name}`,
		 	width: width,
		 	height: height,
		 	alt: name,
		}}),
		
		showHideAnimationType: 'none',
		
		closeOnVerticalDrag: false,
		escKey: false,
		pinchToClose: false,
		clickToCloseNonZoomable: false,

		// zoom
		secondaryZoomLevel: 3,
		maxZoomLevel: 16,

		// disable closing on bg click https://photoswipe.com/click-and-tap-actions/
		bgClickAction: null, 

		pswpModule: () => import('./lib/photoswipe.esm.min.js'),
	}

	const loader = new PhotoSwipeLightbox(options)
	loader.init()

	loader.on('beforeOpen', () => {
		const gallery = loader.pswp

		loader.on('change', () => {
			window.location.hash = gallery.currSlide.data.alt
		})
	})

	function gallery_open() {
		// initial scene determined by url hash
		const url_hash = window.location.hash.substring(1)
		let initial_idx = IMAGE_FILES.indexOf(url_hash)
		if (initial_idx === -1) {
			initial_idx = 0
		}
		loader.loadAndOpen(initial_idx)
	}

	// reopen if it gets closed somehow
	loader.on('close', async (ev) => {
		setInterval(gallery_open, 0.01)
	})

	function open_linked() {

	}

	window.addEventListener('keydown', (ev) => {
		console.log(ev)
		const gallery = loader.pswp

		if (ev.key.toLowerCase() === 'a') {
			const {dset, fid} = gallery.currSlide.data

			const attn_url = `./index_preact.html#${dset}/p2/pair/${fid}`

			console.log(attn_url)
			window.open(attn_url, '_blank')
		}
	})
	gallery_open()
}

// run_gallery(IMAGE_DIR, IMAGE_FILES)
