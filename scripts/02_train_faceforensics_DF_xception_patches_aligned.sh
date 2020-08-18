### training on deepfakes ### 

python train.py --gpu_ids 0 --seed 0 --loadSize 299 --fineSize 299 \
    --name gp2-faceforensics-df --save_epoch_freq 200 \
    --real_im_path dataset/faces/faceforensics_aligned/Deepfakes/original \
    --fake_im_path dataset/faces/faceforensics_aligned/Deepfakes/manipulated \
    --suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience} \
    --which_model_netD xception_block1 --model patch_discriminator \
    --patience 50 --lr_policy constant --max_epochs 1000

python train.py --gpu_ids 0 --seed 0 --loadSize 299 --fineSize 299 \
    --name gp2-faceforensics-df --save_epoch_freq 200 \
    --real_im_path dataset/faces/faceforensics_aligned/Deepfakes/original \
    --fake_im_path dataset/faces/faceforensics_aligned/Deepfakes/manipulated \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience} \
	--which_model_netD xception_block2 --model patch_discriminator \
	--patience 20 --lr_policy constant --max_epochs 1000

python train.py --gpu_ids 0 --seed 0 --loadSize 299 --fineSize 299 \
    --name gp2-faceforensics-df --save_epoch_freq 200 \
    --real_im_path dataset/faces/faceforensics_aligned/Deepfakes/original \
    --fake_im_path dataset/faces/faceforensics_aligned/Deepfakes/manipulated \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience} \
	--which_model_netD xception_block3 --model patch_discriminator \
	--patience 10 --lr_policy constant --max_epochs 1000

python train.py --gpu_ids 0 --seed 0 --loadSize 299 --fineSize 299 \
    --name gp2-faceforensics-df --save_epoch_freq 200 \
    --real_im_path dataset/faces/faceforensics_aligned/Deepfakes/original \
    --fake_im_path dataset/faces/faceforensics_aligned/Deepfakes/manipulated \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience} \
	--which_model_netD xception_block4 --model patch_discriminator \
	--patience 10 --lr_policy constant --max_epochs 1000

python train.py --gpu_ids 0 --seed 0 --loadSize 299 --fineSize 299 \
    --name gp2-faceforensics-df --save_epoch_freq 200 \
    --real_im_path dataset/faces/faceforensics_aligned/Deepfakes/original \
    --fake_im_path dataset/faces/faceforensics_aligned/Deepfakes/manipulated \
	--suffix seed{seed}_{which_model_netD}_{lr_policy}_p{patience} \
	--which_model_netD xception_block5 --model patch_discriminator \
	--patience 10 --lr_policy constant  --max_epochs 1000
