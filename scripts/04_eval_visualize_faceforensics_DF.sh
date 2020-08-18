partition=test

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name  DF \
	--real_im_path dataset/faces/faceforensics_aligned/Deepfakes/original/$partition \
	--fake_im_path dataset/faces/faceforensics_aligned/Deepfakes/manipulated/$partition \
	--train_config checkpoints/gp2-faceforensics-df_seed0_xception_block3_constant_p10/opt.yml

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name NT \
	--real_im_path dataset/faces/faceforensics_aligned/NeuralTextures/original/$partition \
	--fake_im_path dataset/faces/faceforensics_aligned/NeuralTextures/manipulated/$partition \
	--train_config checkpoints/gp2-faceforensics-df_baseline_resnet18_layer1/opt.yml 

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name F2F \
	--real_im_path dataset/faces/faceforensics_aligned/Face2Face/original/$partition \
	--fake_im_path dataset/faces/faceforensics_aligned/Face2Face/manipulated/$partition \
	--train_config checkpoints/gp2-faceforensics-df_baseline_resnet18_layer1/opt.yml 

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name FS \
	--real_im_path dataset/faces/faceforensics_aligned/FaceSwap/original/$partition \
	--fake_im_path dataset/faces/faceforensics_aligned/FaceSwap/manipulated/$partition \
	--train_config checkpoints/gp2-faceforensics-df_seed0_xception_block4_constant_p10/opt.yml 
