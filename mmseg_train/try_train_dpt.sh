
cd /cvlabdata2/home/lis/dev/mtransformer/mmseg_train

# DPT uninitialized
python train_029.py \
	../mtransformer/mmseg_configs/dpt_vit-b16_512x512_160k_cityscapes.py \
	--work-dir /cvlabdata2/home/lis/exp_mmseg/dpt_uninit_ctc \
	--cfg-options model.pretrained=None \

# bash runai_one.sh lis-2040-2dpt 1 "bash mtransformer/mmseg_train/try_train_dpt.sh" 
