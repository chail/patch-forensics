partition=test

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name  DF \
	--real_im_path dataset/faces/faceforensics_aligned/Deepfakes/original/$partition \
	--fake_im_path dataset/faces/faceforensics_aligned/Deepfakes/manipulated/$partition \
	--train_config checkpoints/gp5-faceforensics-f2f_baseline_resnet18_layer1/opt.yml 

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name NT \
	--real_im_path dataset/faces/faceforensics_aligned/NeuralTextures/original/$partition \
	--fake_im_path dataset/faces/faceforensics_aligned/NeuralTextures/manipulated/$partition \
	--train_config checkpoints/gp5-faceforensics-f2f_seed0_xception_block1_constant_p50/opt.yml 

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name F2F \
	--real_im_path dataset/faces/faceforensics_aligned/Face2Face/original/$partition \
	--fake_im_path dataset/faces/faceforensics_aligned/Face2Face/manipulated/$partition \
	--train_config checkpoints/gp5-faceforensics-f2f_baseline_resnet18_layer1/opt.yml 

python test.py --which_epoch bestval --gpu_ids 0 --partition $partition \
	--visualize --average_mode after_softmax --topn 100 --force_redo \
	--dataset_name FS \
	--real_im_path dataset/faces/faceforensics_aligned/FaceSwap/original/$partition \
	--fake_im_path dataset/faces/faceforensics_aligned/FaceSwap/manipulated/$partition \
	--train_config checkpoints/gp5-faceforensics-f2f_seed0_xception_block2_constant_p20/opt.yml 
