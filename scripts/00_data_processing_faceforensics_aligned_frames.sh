### Deepfakes processing ###

# train 
python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated resources/faceforensics_raw/manipulated_sequences/Deepfakes/c23/videos \
	--source_dir_original resources/faceforensics_raw/original_sequences/youtube/c23/videos \
	--outsize 128 --split resources/faceforensics_raw/train.json \
	--output_dir dataset/faces/faceforensics_aligned/Deepfakes

# val
python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated resources/faceforensics_raw/manipulated_sequences/Deepfakes/c23/videos \
	--source_dir_original resources/faceforensics_raw/original_sequences/youtube/c23/videos \
	--outsize 128 --split resources/faceforensics_raw/val.json \
	--output_dir dataset/faces/faceforensics_aligned/Deepfakes

# test
python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated resources/faceforensics_raw/manipulated_sequences/Deepfakes/c23/videos \
	--source_dir_original resources/faceforensics_raw/original_sequences/youtube/c23/videos \
	--outsize 128 --split resources/faceforensics_raw/test.json \
	--output_dir dataset/faces/faceforensics_aligned/Deepfakes

### Neural Textures processing ###

# train 
python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated resources/faceforensics_raw/manipulated_sequences/NeuralTextures/c23/videos \
	--source_dir_original resources/faceforensics_raw/original_sequences/youtube/c23/videos \
	--outsize 128 --split resources/faceforensics_raw/train.json \
	--output_dir dataset/faces/faceforensics_aligned/NeuralTextures

# val
python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated resources/faceforensics_raw/manipulated_sequences/NeuralTextures/c23/videos \
	--source_dir_original resources/faceforensics_raw/original_sequences/youtube/c23/videos \
	--outsize 128 --split resources/faceforensics_raw/val.json \
	--output_dir dataset/faces/faceforensics_aligned/NeuralTextures

# test
python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated resources/faceforensics_raw/manipulated_sequences/NeuralTextures/c23/videos \
	--source_dir_original resources/faceforensics_raw/original_sequences/youtube/c23/videos \
	--outsize 128 --split resources/faceforensics_raw/test.json \
	--output_dir dataset/faces/faceforensics_aligned/NeuralTextures

### Face2Face processing ###

# train 
python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated resources/faceforensics_raw/manipulated_sequences/Face2Face/c23/videos \
	--source_dir_original resources/faceforensics_raw/original_sequences/youtube/c23/videos \
	--outsize 128 --split resources/faceforensics_raw/train.json \
	--output_dir dataset/faces/faceforensics_aligned/Face2Face

# val
python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated resources/faceforensics_raw/manipulated_sequences/Face2Face/c23/videos \
	--source_dir_original resources/faceforensics_raw/original_sequences/youtube/c23/videos \
	--outsize 128 --split resources/faceforensics_raw/val.json \
	--output_dir dataset/faces/faceforensics_aligned/Face2Face

# test
python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated resources/faceforensics_raw/manipulated_sequences/Face2Face/c23/videos \
	--source_dir_original resources/faceforensics_raw/original_sequences/youtube/c23/videos \
	--outsize 128 --split resources/faceforensics_raw/test.json \
	--output_dir dataset/faces/faceforensics_aligned/Face2Face

### FaceSwap processing ###

# train 
python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated resources/faceforensics_raw/manipulated_sequences/FaceSwap/c23/videos \
	--source_dir_original resources/faceforensics_raw/original_sequences/youtube/c23/videos \
	--outsize 128 --split resources/faceforensics_raw/train.json \
	--output_dir dataset/faces/faceforensics_aligned/FaceSwap

# val
python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated resources/faceforensics_raw/manipulated_sequences/FaceSwap/c23/videos \
	--source_dir_original resources/faceforensics_raw/original_sequences/youtube/c23/videos \
	--outsize 128 --split resources/faceforensics_raw/val.json \
	--output_dir dataset/faces/faceforensics_aligned/FaceSwap

# test
python -m data.processing.faceforensics_process_frames \
	--source_dir_manipulated resources/faceforensics_raw/manipulated_sequences/FaceSwap/c23/videos \
	--source_dir_original resources/faceforensics_raw/original_sequences/youtube/c23/videos \
	--outsize 128 --split resources/faceforensics_raw/test.json \
	--output_dir dataset/faces/faceforensics_aligned/FaceSwap
