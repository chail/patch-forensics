#!/bin/bash

OUTPUT_ROOT=dataset/faces/ffhq

python -m data.processing.sgan_tf \
	--num_samples 5000 --seed 1 --batch_size 8 --gpu 0 \
	--pretrained ffhq --format png --resize 128 \
	--output_path $OUTPUT_ROOT/sgan-pretrained-128-png/test

python -m data.processing.sgan_tf \
	--num_samples 5000 --seed 2 --batch_size 8 --gpu 0 \
	--pretrained ffhq --format png --resize 128 \
	--output_path $OUTPUT_ROOT/sgan-pretrained-128-png/val
