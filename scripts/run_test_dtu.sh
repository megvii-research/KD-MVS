#!/usr/bin/env bash
# run this script in the root path of KD-MVS
TESTPATH="/data/DTU/dtu-test-processed" # path to dataset dtu_test
TESTLIST="lists/dtu/test.txt"
CKPT_FILE="ckpt/model_kd_dtu.ckpt" # path to checkpoint file
FUSIBILE_EXE="../fusibile/fusibile" # path to gipuma fusible file
OUTDIR="outputs/test_dtu" # path to save outputs
if [ ! -d $OUTDIR ]; then
	mkdir -p $OUTDIR
fi

python test.py \
--dataset=general_eval \
--batch_size=1 \
--testpath=$TESTPATH  \
--testlist=$TESTLIST \
--loadckpt=$CKPT_FILE \
--outdir=$OUTDIR \
--fusibile_exe_path=$FUSIBILE_EXE \
--interval_scale=1.06