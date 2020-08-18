# pretrained pgan with inversions
python train.py --gpu_ids 0 --seed 0 --loadSize 224 --fineSize 224 \
	--name gp1-gan-winversion --save_epoch_freq 200 \
	--real_im_path dataset/faces/celebahq/inverted_and_unpaired_real \
	--fake_im_path dataset/faces/celebahq/inverted_and_unpaired_fake \
	--suffix baseline_resnet18_full \
	--which_model_netD resnet18 --model basic_discriminator \
	--patience 10 --lr_policy constant --max_epochs 1000

# faceforensics DF
python train.py --gpu_ids 0 --seed 0 --loadSize 224 --fineSize 224 \
	--name gp2-faceforensics-df --save_epoch_freq 200 \
	--real_im_path dataset/faces/faceforensics_aligned/Deepfakes/original \
	--fake_im_path dataset/faces/faceforensics_aligned/Deepfakes/manipulated \
	--suffix baseline_resnet18_full \
	--which_model_netD resnet18 --model basic_discriminator \
	--patience 10 --lr_policy constant --max_epochs 1000

# faceforensics NT
python train.py --gpu_ids 0 --seed 0 --loadSize 224 --fineSize 224 \
	--name gp3-faceforensics-nt --save_epoch_freq 200 \
	--real_im_path dataset/faces/faceforensics_aligned/NeuralTextures/original \
	--fake_im_path dataset/faces/faceforensics_aligned/NeuralTextures/manipulated \
	--suffix baseline_resnet18_full \
	--which_model_netD resnet18 --model basic_discriminator \
	--patience 10 --lr_policy constant --max_epochs 1000

# faceforensics FS
python train.py --gpu_ids 0 --seed 0 --loadSize 224 --fineSize 224 \
	--name gp4-faceforensics-fs --save_epoch_freq 200 \
	--real_im_path dataset/faces/faceforensics_aligned/FaceSwap/original \
	--fake_im_path dataset/faces/faceforensics_aligned/FaceSwap/manipulated \
	--suffix baseline_resnet18_full \
	--which_model_netD resnet18 --model basic_discriminator \
	--patience 10 --lr_policy constant --max_epochs 1000

# faceforensics F2F
python train.py --gpu_ids 0 --seed 0 --loadSize 224 --fineSize 224 \
	--name gp5-faceforensics-f2f --save_epoch_freq 200 \
	--real_im_path dataset/faces/faceforensics_aligned/Face2Face/original \
	--fake_im_path dataset/faces/faceforensics_aligned/Face2Face/manipulated \
	--suffix baseline_resnet18_full \
	--which_model_netD resnet18 --model basic_discriminator \
	--patience 10 --lr_policy constant --max_epochs 1000

