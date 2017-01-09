#!/bin/bash
DATA='JHMDB'
MAX_EPOCHS=100
STEPS_CKPTS=10
for NUM_UNITS in 8 16 32 64
do
    for BATCH_SIZE in 8 16 32 64 128
    do
	for NUM_FRAMES in 10
	do
	    python action_recognition_srnn.py --num_units $NUM_UNITS --batch_size $BATCH_SIZE --max_epochs $MAX_EPOCHS --num_frames $NUM_FRAMES --train_dir 'models/'$DATA'_normalized_'$NUM_UNITS'units_'$BATCH_SIZE'batch_'$NUM_FRAMES'frames' --gpu '/gpu:3'
	done
    done
done
