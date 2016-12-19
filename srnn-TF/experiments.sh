#!/bin/bash
DATA='JHMDB'
MAX_EPOCHS=100
STEPS_CKPTS=10
for NUM_UNITS in 4 8 16 32 64 128
do
    for BATCH_SIZE in 4 8 16 32 64
    do
	for NUM_FRAMES in 40 20 10
	do
	    python action_recognition_srnn.py --num_units $NUM_UNITS --batch_size $BATCH_SIZE --max_epochs $MAX_EPOCHS --num_frames $NUM_FRAMES --train_dir 'models/'$DATA'_'$NUM_UNITS'units_'$BATCH_SIZE'batch_'$NUM_FRAMES'frames'
	done
    done
done
