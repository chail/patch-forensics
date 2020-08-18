partition=test

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name celebahq-pgan-pretrained \
	--real_im_path dataset/faces/celebahq/real-tfr-1024-resized128/$partition \
	--fake_im_path dataset/faces/celebahq/pgan-pretrained-128-png/$partition \
	--train_config checkpoints/gp1-gan-winversion_seed0_xception_block2_constant_p20/opt.yml 

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name celebahq-sgan-pretrained \
	--real_im_path dataset/faces/celebahq/real-tfr-1024-resized128/$partition \
	--fake_im_path dataset/faces/celebahq/sgan-pretrained-128-png/$partition \
	--train_config checkpoints/gp1-gan-winversion_seed0_xception_block3_constant_p10/opt.yml 

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name celebahq-glow-pretrained \
	--real_im_path dataset/faces/celebahq/real-tfr-1024-resized128/$partition \
	--fake_im_path dataset/faces/celebahq/glow-pretrained-128-png/$partition \
	--train_config checkpoints/gp1d-gan-samplesonly_seed0_xception_block1_constant_p50/opt.yml 

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name celeba-gmm \
	--real_im_path dataset/faces/celeba/mfa-real/$partition \
	--fake_im_path dataset/faces/celeba/mfa-defaults/$partition \
	--train_config checkpoints/gp1-gan-winversion_seed0_xception_block2_constant_p20/opt.yml 

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name ffhq-pgan \
	--real_im_path dataset/faces/ffhq/real-tfr-1024-resized128/$partition \
	--fake_im_path dataset/faces/ffhq/pgan-9k-128-png/$partition \
	--train_config checkpoints/gp1-gan-winversion_seed0_xception_block2_constant_p20/opt.yml 

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name ffhq-sgan \
	--real_im_path dataset/faces/ffhq/real-tfr-1024-resized128/$partition \
	--fake_im_path dataset/faces/ffhq/sgan-pretrained-128-png/$partition \
	--train_config checkpoints/gp1-gan-winversion_seed0_xception_block4_constant_p10/opt.yml 

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name ffhq-sgan2 \
	--real_im_path dataset/faces/ffhq/real-tfr-1024-resized128/$partition \
	--fake_im_path dataset/faces/ffhq/sgan2-pretrained-128-png/$partition \
	--train_config checkpoints/gp1-gan-winversion_seed0_xception_block3_constant_p10/opt.yml 
