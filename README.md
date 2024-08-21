
# AttEntropy: Segmenting Unknown Objects in Complex Scenes using the Spatial Attention Entropy of Semantic Segmentation Transformers

```bash
pip install easydict openmim 
# mim install mmcv
# pip install mmsegmentation==0.30.0 mmpretrain
mim install --upgrade mmsegmentation mmpretrain
```

Download an example image
```bash
mkdir demo
wget https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Beeston_MMB_A6_Middle_Street.jpg/640px-Beeston_MMB_A6_Middle_Street.jpg -O demo/Beeston_MMB_A6_Middle_Street.jpg
```

Infer on the example image
```bash
python -m mtransformer.evaluation heatmaps-local --method SETR-AttnEntropy_auto12 --local-imgs demo/Beeston_MMB_A6_Middle_Street.jpg
```