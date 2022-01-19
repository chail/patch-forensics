# stylegan
git clone https://github.com/NVlabs/stylegan.git 

# progressive gan
git clone https://github.com/tkarras/progressive_growing_of_gans.git 

# glow -- note: will also need to download glow pretrained weights from 
# glow/demo/script.sh
git clone https://github.com/openai/glow.git 

# celebahq progressive gan
gdown https://drive.google.com/uc?id=188K19ucknC6wg1R6jbuPEhTq9zoufOx4

# celebahq image indices: original source from https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ
# (previous)
# gdown https://drive.google.com/uc?id=0B4qLcYyJmiz0U25vdEVIU3NvNFk
# mv image_list.txt celebahq_image_list.txt
# (alternate link)
wget http://latent-composition.csail.mit.edu/other_projects/patch_forensics/resources/celebahq_image_list.txt

# celeba train/test/val partitions 
# from celeba google drive -> eval
# (previous)
# gdown https://drive.google.com/uc?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk
# mv list_eval_partition.txt celeba_list_eval_partition.txt
# (alternate link) 
wget http://latent-composition.csail.mit.edu/other_projects/patch_forensics/resources/celeba_list_eval_partition.txt

# dlib facial landmarks predictor
wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# face parsing pytorch repository - rename it
git clone https://github.com/zllrunning/face-parsing.PyTorch.git face_parsing_pytorch
# also need to download the weights for the face parser; following the steps from that repo
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812  # pretrained model
mkdir -p face_parsing_pytorch/res/cp
mv 79999_iter.pth face_parsing_pytorch/res/cp
