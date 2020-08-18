# script to run top patches experiments
# model: trained on deepfake
partition=test

# test on DF
ckpt=gp2-faceforensics-df_seed0_xception_block3_constant_p10
name=DF
python patches.py --which_epoch bestval --gpu_ids 0 \
	--topn 10000 --unique --partition $partition \
	--dataset_name $name \
	--train_config checkpoints/$ckpt/opt.yml \
	--real_im_path dataset/faces/faceforensics_aligned/Deepfakes/original/$partition/ \
       	--fake_im_path dataset/faces/faceforensics_aligned/Deepfakes/manipulated/$partition/
python segmenter.py results/$ckpt/$partition/epoch_bestval/$name/patches_top10000/

# test on NT
ckpt=gp2-faceforensics-df_baseline_resnet18_layer1
name=NT
python patches.py --which_epoch bestval --gpu_ids 0 \
	--topn 10000 --unique --partition $partition \
	--dataset_name $name \
	--train_config checkpoints/$ckpt/opt.yml \
	--real_im_path dataset/faces/faceforensics_aligned/NeuralTextures/original/$partition/ \
       	--fake_im_path dataset/faces/faceforensics_aligned/NeuralTextures/manipulated/$partition/ 
python segmenter.py results/$ckpt/$partition/epoch_bestval/$name/patches_top10000

# test on F2F
ckpt=gp2-faceforensics-df_baseline_resnet18_layer1
name=F2F
python patches.py --which_epoch bestval --gpu_ids 0 \
	--topn 10000 --unique --partition $partition \
	--dataset_name $name \
	--train_config checkpoints/$ckpt/opt.yml \
	--real_im_path dataset/faces/faceforensics_aligned/Face2Face/original/$partition/ \
       	--fake_im_path dataset/faces/faceforensics_aligned/Face2Face/manipulated/$partition/ 
python segmenter.py results/$ckpt/$partition/epoch_bestval/$name/patches_top10000


# test on FS
ckpt=gp2-faceforensics-df_seed0_xception_block4_constant_p10
name=FS
python patches.py --which_epoch bestval --gpu_ids 0 \
	--topn 10000 --unique --partition $partition \
	--dataset_name $name \
	--train_config checkpoints/$ckpt/opt.yml \
	--real_im_path dataset/faces/faceforensics_aligned/FaceSwap/original/$partition/ \
       	--fake_im_path dataset/faces/faceforensics_aligned/FaceSwap/manipulated/$partition/ 
python segmenter.py results/$ckpt/$partition/epoch_bestval/$name/patches_top10000
