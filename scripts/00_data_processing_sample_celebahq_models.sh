#!/bin/bash

CELEBAHQ_TRAIN=24183
CELEBAHQ_VAL=2993 
CELEBAHQ_TEST=2824

OUTPUT_ROOT=dataset/faces/celebahq

### PGAN CELEBAHQ SAMPLES ### 
PGAN_MODEL=resources/karras2018iclr-celebahq-1024x1024.pkl

# pgan train
python -m data.processing.pgan_tf \
       	--num_samples $CELEBAHQ_TRAIN --seed 0 --batch_size 32 --gpu 0 \
	--model_path $PGAN_MODEL --format png --resize 128 \
	--output_path $OUTPUT_ROOT/pgan-pretrained-128-png/train

# pgan test
python -m data.processing.pgan_tf \
	--num_samples $CELEBAHQ_TEST --seed 1 --batch_size 32 --gpu 0 \
	--model_path $PGAN_MODEL --format png --resize 128 \
	--output_path $OUTPUT_ROOT/pgan-pretrained-128-png/test

# pgan val
python -m data.processing.pgan_tf \
	--num_samples $CELEBAHQ_VAL --seed 2 --batch_size 32 --gpu 0 \
	--model_path $PGAN_MODEL --format png --resize 128 \
	--output_path $OUTPUT_ROOT/pgan-pretrained-128-png/val

# SGAN CELEBAHQ SAMPLES ###

# sgan train
python -m data.processing.sgan_tf \
	--num_samples $CELEBAHQ_TRAIN --seed 0 --batch_size 32 --gpu 0 \
	--pretrained celebahq --format png --resize 128 \
	--output_path $OUTPUT_ROOT/sgan-pretrained-128-png/train

# sgan test
python -m data.processing.sgan_tf \
	--num_samples $CELEBAHQ_TEST --seed 1 --batch_size 32 --gpu 0 \
	--pretrained celebahq --format png --resize 128 \
	--output_path $OUTPUT_ROOT/sgan-pretrained-128-png/test

# sgan val 
python -m data.processing.sgan_tf \
	--num_samples $CELEBAHQ_VAL --seed 2 --batch_size 32 --gpu 0 \
	--pretrained celebahq --format png --resize 128 \
	--output_path $OUTPUT_ROOT/sgan-pretrained-128-png/val

### GLOW CELEBAHQ SAMPLES ###

# glow train
python -m data.processing.glow_tf \
	--num_samples $CELEBAHQ_TRAIN --seed 0 --batch_size 16 --gpu 0 \
	--manipulate --pretrained celebahq --format png --resize 128 \
	--output_path $OUTPUT_ROOT/glow-pretrained-128-png/train

# glow test
python -m data.processing.glow_tf \
	--num_samples $CELEBAHQ_TEST --seed 1 --batch_size 16 --gpu 0 \
	--manipulate --pretrained celebahq --format png --resize 128 \
	--output_path $OUTPUT_ROOT/glow-pretrained-128-png/test

# glow val
python -m data.processing.glow_tf \
	--num_samples $CELEBAHQ_VAL --seed 2 --batch_size 16 --gpu 0 \
	--manipulate --pretrained celebahq --format png --resize 128 \
	--output_path $OUTPUT_ROOT/glow-pretrained-128-png/val

