# stylegan
git clone https://github.com/NVlabs/stylegan.git 

# progressive gan
git clone https://github.com/tkarras/progressive_growing_of_gans.git 

# glow -- note: will also need to download glow pretrained weights from 
# glow/demo/script.sh
git clone https://github.com/openai/glow.git 

# celebahq progressive gan
gdown https://drive.google.com/uc?id=188K19ucknC6wg1R6jbuPEhTq9zoufOx4

# celebahq image indices
gdown https://drive.google.com/uc?id=0B4qLcYyJmiz0U25vdEVIU3NvNFk
mv image_list.txt celebahq_image_list.txt

# celeba train/test/val partitions 
# from celeba google drive -> eval
gdown https://drive.google.com/uc?id=0B7EVK8r0v71pY0NSMzRuSXJEVkk
mv list_eval_partition.txt celeba_list_eval_partition.txt

# dlib facial landmarks predictor
wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2


# face parsing pytorch repository - rename properly
# also need to download the weights for the face parser
