
# AttEntropy: Segmenting Unknown Objects in Complex Scenes using the Spatial Attention Entropy of Semantic Segmentation Transformers


Download an example image
```bash
mkdir demo
https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Beeston_MMB_A6_Middle_Street.jpg/640px-Beeston_MMB_A6_Middle_Street.jpg -O demo/Beeston_MMB_A6_Middle_Street.jpg
```



```bash

python -m mtransformer.evaluation heatmaps-local --method SETR-AttnEntropy_auto12 --local-imgs demo/Beeston_MMB_A6_Middle_Street.jpg

```