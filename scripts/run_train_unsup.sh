# Run this script in the root path of KD-MVS
DATASET_DIR="/data/DTU/mvs_training/dtu/"   # path to dataset folder
TESTLIST="lists/dtu/train.txt"
LOG_DIR="outputs/train_unsup"               # path to save the log file
if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi

NGPUS=8
BATCH_SIZE=1
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_unsup.py \
--logdir=$LOG_DIR \
--dataset=dtu_ut \
--batch_size=$BATCH_SIZE \
--epochs=10 \
--trainpath=$DATASET_DIR \
--trainlist=lists/dtu/train.txt \
--testlist=lists/dtu/test.txt \
--numdepth=192 \
--ndepths="48,32,8" \
--nviews=5 \
--wd=0.0001 \
--depth_inter_r="4.0,1.0,0.5" \
--lrepochs="6,8,12:2" \
--dlossw="1.0,1.0,1.0" | tee -a $LOG_DIR/log.txt