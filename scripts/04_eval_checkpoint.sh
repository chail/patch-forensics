# example: python test_runs.py checkpoints/temp-inverted-and-unpaired_seed0_xception_block5_constant_p10/ gen_models val

# model arch and dataset
for expt in checkpoints/gp1-* checkpoints/gp1[a-d]-*
do
for part in test # val
do
    cmd="python test_runs.py $expt gen_models $part"
    echo $cmd
    eval $cmd
done
done

# # faceforensics -- this will only run if the faceforensics
# # dataset is processed according to 
# # scripts/00_data_processing_faceforensics_aligned_frames.sh
# for expt in checkpoints/gp[2-5]-* 
# do
# for part in val test
# do
#     cmd="python test_runs.py $expt faceforensics $part"
#     echo $cmd
#     eval $cmd
# done
# done

