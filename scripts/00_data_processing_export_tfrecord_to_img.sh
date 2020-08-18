python -m data.processing.export_tfrecord_to_img \
	--tfrecord resources/tfrecords/celebahq/celeba-hq-r10.tfrecords \
	--outdir dataset/faces/celebahq/real-tfr-1024-resized128 \
	--outsize 128 --dataset celebahq

python -m data.processing.export_tfrecord_to_img \
	--tfrecord resources/tfrecords/ffhq/ffhq-r10.tfrecords \
	--outdir dataset/faces/ffhq/real-tfr-1024-resized-128 \
	--outsize 128 --dataset ffhq

