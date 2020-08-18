# script to run top patches experiments

partition=test

# pgan pretrain
ckpt=gp1-gan-winversion_seed0_xception_block2_constant_p20
name=celebahq-pgan-pretrained
python patches.py --which_epoch bestval --gpu_ids 0 \
	--topn 10000 --unique --partition $partition \
	--train_config checkpoints/$ckpt/opt.yml \
	--dataset_name $name \
	--real_im_path dataset/faces/celebahq/real-tfr-1024-resized128/$partition/ \
       	--fake_im_path dataset/faces/celebahq/pgan-pretrained-128-png/$partition/
python segmenter.py results/$ckpt/$partition/epoch_bestval/$name/patches_top10000/

# sgan pretrain
ckpt=gp1-gan-winversion_seed0_xception_block3_constant_p10
name=celebahq-sgan-pretrained
python patches.py --which_epoch bestval --gpu_ids 0 \
	--topn 10000 --unique --partition $partition \
	--train_config checkpoints/$ckpt/opt.yml \
	--dataset_name $name \
	--real_im_path dataset/faces/celebahq/real-tfr-1024-resized128/$partition \
	--fake_im_path dataset/faces/celebahq/sgan-pretrained-128-png/$partition
python segmenter.py results/$ckpt/$partition/epoch_bestval/$name/patches_top10000/

# glow pretrain
ckpt=gp1d-gan-samplesonly_seed0_xception_block1_constant_p50
name=celebahq-glow-pretrained
python patches.py --which_epoch bestval --gpu_ids 0 \
	--topn 10000 --unique --partition $partition \
	--train_config checkpoints/$ckpt/opt.yml \
	--dataset_name $name \
	--real_im_path dataset/faces/celebahq/real-tfr-1024-resized128/$partition \
	--fake_im_path dataset/faces/celebahq/glow-pretrained-128-png/$partition
python segmenter.py results/$ckpt/$partition/epoch_bestval/$name/patches_top10000/

# gmm model
ckpt=gp1-gan-winversion_seed0_xception_block2_constant_p20
name=celeba-gmm
python patches.py --which_epoch bestval --gpu_ids 0 \
	--topn 10000 --unique --partition $partition \
	--train_config checkpoints/$ckpt/opt.yml \
	--dataset_name $name \
	--real_im_path dataset/faces/celeba/mfa-real/$partition \
	--fake_im_path dataset/faces/celeba/mfa-defaults/$partition
python segmenter.py results/$ckpt/$partition/epoch_bestval/$name/patches_top10000/

# ffhq pgan
ckpt=gp1-gan-winversion_seed0_xception_block2_constant_p20
name=ffhq-pgan
python patches.py --which_epoch bestval --gpu_ids 0 \
	--topn 10000 --unique --partition $partition \
	--train_config checkpoints/$ckpt/opt.yml \
	--dataset_name $name \
	--real_im_path dataset/faces/ffhq/real-tfr-1024-resized128/$partition \
	--fake_im_path dataset/faces/ffhq/pgan-9k-128-png/$partition
python segmenter.py results/$ckpt/$partition/epoch_bestval/$name/patches_top10000/

# ffhq sgan
ckpt=gp1-gan-winversion_seed0_xception_block4_constant_p10
name=ffhq-sgan
python patches.py --which_epoch bestval --gpu_ids 0 \
	--topn 10000 --unique --partition $partition \
	--train_config checkpoints/$ckpt/opt.yml \
	--dataset_name $name \
	--real_im_path dataset/faces/ffhq/real-tfr-1024-resized128/$partition \
	--fake_im_path dataset/faces/ffhq/sgan-pretrained-128-png/$partition
python segmenter.py results/$ckpt/$partition/epoch_bestval/$name/patches_top10000/

# ffhq sgan2
ckpt=gp1-gan-winversion_seed0_xception_block3_constant_p10
name=ffhq-sgan2
python patches.py --which_epoch bestval --gpu_ids 0 \
	--topn 10000 --unique --partition $partition \
	--train_config checkpoints/$ckpt/opt.yml \
	--dataset_name $name \
	--real_im_path dataset/faces/ffhq/real-tfr-1024-resized128/$partition \
	--fake_im_path dataset/faces/ffhq/sgan2-pretrained-128-png/$partition
python segmenter.py results/$ckpt/$partition/epoch_bestval/$name/patches_top10000/
