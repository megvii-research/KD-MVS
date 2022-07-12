#!/usr/bin/env bash
# run this script in the root path of KD-MVS
TESTPATH="/data/DTU/dtu-test" # path to dataset dtu_test
TESTLIST="lists/dtu/test.txt"
CKPT_FILE="ckpt/model_kd_dtu.ckpt" # path to checkpoint file
# uncomment the fallowing line if you choose to use gipuma fusion method
# FUSIBILE_EXE="/data/Trans_cas/fusibile/fusibile"
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
--numdepth=192 \
--ndepths="48,32,8" \
--depth_inter_r="4.0,1.0,0.5" \
--interval_scale=1.06 \
--filter_method="normal"

# --filter_method="gipuma" \
# --fusibile_exe_path=$FUSIBILE_EXE \
# --conf=0.01

