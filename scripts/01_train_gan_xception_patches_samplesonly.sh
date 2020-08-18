### celebahq-pgan raw samples from the gan vs real images ###
### uses --no_serial_batches as samples are not paired ### 
python train.py --gpu_ids 0 --seed 0 --loadSize 299 --fineSize 299 \
	--name gp1d-gan-samplesonly --save_epoch_freq 200 \
 	--real_im_path dataset/faces/celebahq/real-tfr-1024-resized128 \
 	--fake_im_path dataset/faces/celebahq/pgan-pretrained-128-png \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience} \
	--which_model_netD xception_block1 --model patch_discriminator \
	--patience 50 --lr_policy constant --max_epochs 1000 \
	--no_serial_batches

python train.py --gpu_ids 0 --seed 0 --loadSize 299 --fineSize 299 \
	--name gp1d-gan-samplesonly --save_epoch_freq 200 \
 	--real_im_path dataset/faces/celebahq/real-tfr-1024-resized128 \
 	--fake_im_path dataset/faces/celebahq/pgan-pretrained-128-png \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience} \
	--which_model_netD xception_block2 --model patch_discriminator \
	--patience 20 --lr_policy constant --max_epochs 1000 \
	--no_serial_batches

python train.py --gpu_ids 0 --seed 0 --loadSize 299 --fineSize 299 \
	--name gp1d-gan-samplesonly --save_epoch_freq 50 \
 	--real_im_path dataset/faces/celebahq/real-tfr-1024-resized128 \
 	--fake_im_path dataset/faces/celebahq/pgan-pretrained-128-png \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience} \
	--which_model_netD xception_block3 --model patch_discriminator \
	--patience 10 --lr_policy constant --max_epochs 1000 \
	--no_serial_batches

python train.py --gpu_ids 0 --seed 0 --loadSize 299 --fineSize 299 \
	--name gp1d-gan-samplesonly --save_epoch_freq 50 \
 	--real_im_path dataset/faces/celebahq/real-tfr-1024-resized128 \
 	--fake_im_path dataset/faces/celebahq/pgan-pretrained-128-png \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience} \
	--which_model_netD xception_block4 --model patch_discriminator \
	--patience 10 --lr_policy constant --max_epochs 1000 \
	--no_serial_batches

python train.py --gpu_ids 0 --seed 0 --loadSize 299 --fineSize 299 \
	--name gp1d-gan-samplesonly --save_epoch_freq 50 \
 	--real_im_path dataset/faces/celebahq/real-tfr-1024-resized128 \
 	--fake_im_path dataset/faces/celebahq/pgan-pretrained-128-png \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience} \
	--which_model_netD xception_block5 --model patch_discriminator \
	--patience 10 --lr_policy constant  --max_epochs 1000 \
	--no_serial_batches
