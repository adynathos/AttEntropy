
cd /cvlabdata2/home/lis/dev/mtransformer/mmseg_train

# SETR uninitialized
python train_029.py \
	../mtransformer/mmseg_configs/setr_vit-large_pup_8x1_768x768_80k_cityscapes.py \
	--work-dir /cvlabdata2/home/lis/exp_mmseg/setr_uninit_ctc \
	--cfg-options model.backbone.init_cfg=None \

# bash runai_one.sh lis-2040-1setr 1 "bash mtransformer/mmseg_train/try_train_setr.sh" 