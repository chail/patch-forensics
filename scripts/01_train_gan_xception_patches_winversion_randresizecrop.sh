### celebahq-pgan generated faces and inverted real pairs ###
python train.py --gpu_ids 0 --seed 0 --loadSize 333 --fineSize 299 \
	--name gp1a-gan-winversion --save_epoch_freq 200 \
	--real_im_path dataset/faces/celebahq/inverted_and_unpaired_real \
	--fake_im_path dataset/faces/celebahq/inverted_and_unpaired_fake \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience}_randresizecrop \
	--which_model_netD xception_block1 --model patch_discriminator \
	--patience 50 --lr_policy constant --max_epochs 1000 --random_resized_crop

python train.py --gpu_ids 0 --seed 0 --loadSize 333 --fineSize 299 \
	--name gp1a-gan-winversion --save_epoch_freq 200 \
	--real_im_path dataset/faces/celebahq/inverted_and_unpaired_real \
	--fake_im_path dataset/faces/celebahq/inverted_and_unpaired_fake \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience}_randresizecrop \
	--which_model_netD xception_block2 --model patch_discriminator \
	--patience 20 --lr_policy constant --max_epochs 1000 --random_resized_crop

python train.py --gpu_ids 0 --seed 0 --loadSize 333 --fineSize 299 \
	--name gp1a-gan-winversion --save_epoch_freq 50 \
	--real_im_path dataset/faces/celebahq/inverted_and_unpaired_real \
	--fake_im_path dataset/faces/celebahq/inverted_and_unpaired_fake \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience}_randresizecrop \
	--which_model_netD xception_block3 --model patch_discriminator \
	--patience 10 --lr_policy constant --max_epochs 1000 --random_resized_crop

python train.py --gpu_ids 0 --seed 0 --loadSize 333 --fineSize 299 \
	--name gp1a-gan-winversion --save_epoch_freq 50 \
	--real_im_path dataset/faces/celebahq/inverted_and_unpaired_real \
	--fake_im_path dataset/faces/celebahq/inverted_and_unpaired_fake \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience}_randresizecrop \
	--which_model_netD xception_block4 --model patch_discriminator \
	--patience 10 --lr_policy constant --max_epochs 1000 --random_resized_crop

python train.py --gpu_ids 0 --seed 0 --loadSize 333 --fineSize 299 \
	--name gp1a-gan-winversion --save_epoch_freq 50 \
	--real_im_path dataset/faces/celebahq/inverted_and_unpaired_real \
	--fake_im_path dataset/faces/celebahq/inverted_and_unpaired_fake \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience}_randresizecrop \
	--which_model_netD xception_block5 --model patch_discriminator \
	--patience 10 --lr_policy constant  --max_epochs 1000 --random_resized_crop
