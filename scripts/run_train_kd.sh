DATASET_DIR="/data/DTU/mvs_training/dtu/"	# path to dataset folder
TESTLIST="lists/dtu/train.txt"
LOAD_CKPT_DIR="ckpt/model_unsup_dtu.ckpt"	# path to checkpoint of the teacher model
LOG_DIR="outputs"							# path to save the log file
FUSIBILE_EXE=""	# path to fusible file of gipuma
EST_DEPTH_DIR="$LOG_DIR/outputs_tescher_model/est_depth/"
PSEUDO_PC_DIR="$LOG_DIR/outputs_tescher_model/pseudo_pc/"
CHECKED_DEPTH_DIR="$LOG_DIR/outputs_tescher_model/checked_depth/"
NGPUS=8
BATCH_SIZE=2

if [ ! -d $PSEUDO_PC_DIR ]; then
	mkdir -p $PSEUDO_PC_DIR
fi
if [ ! -d $EST_DEPTH_DIR ]; then
	mkdir -p $EST_DEPTH_DIR
fi
if [ ! -d $CHECKED_DEPTH_DIR ]; then
	mkdir -p $CHECKED_DEPTH_DIR
fi


### Step 1: Inference on training set#####
# python infer_dtu_trainset.py \
# --dataset=dtu_yao_test \
# --batch_size=$BATCH_SIZE \
# --testpath=$DATASET_DIR  \
# --pcdpath=$PSEUDO_PC_DIR \
# --testlist=$TESTLIST \
# --loadckpt=$LOAD_CKPT_DIR \
# --outdir=$EST_DEPTH_DIR \
# --numdepth=192 \
# --ndepths="48,32,8" \
# --depth_inter_r="4.0,1.0,0.5" \
# --interval_scale=1.06 \
# --filter_method="dynamic"


#### Step 2: Dynamic check & Prob encoding #####
# Note the following cross_check_prob_enc_dtu.py is a simple version of implementation
# python tools/cross_check_prob_enc_dtu.py \
# --testpath=$EST_DEPTH_DIR \
# --pairpath=$DATASET_DIR \
# --pseudo_depth_path=$EST_DEPTH_DIR \
# --testlist=$TESTLIST \
# --outdir=$CHECKED_DEPTH_DIR \
# --photo_threshold=0.05


##### Step 3: KD training #####
BATCH_SIZE=1
LOG_DIRS="$LOG_DIR/student_kd_train/"
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_kd.py \
	--logdir=$LOG_DIRS \
	--dataset=dtu_kd \
	--batch_size=$BATCH_SIZE \
	--epochs=20 \
	--trainpath=$DATASET_DIR \
	--pseudopath=$CHECKED_DEPTH_DIR \
	--trainlist=lists/dtu/train.txt \
    --testlist=lists/dtu/test.txt \
	--numdepth=192 \
    --ndepths="48,32,8" \
	--nviews=5 \
	--wd=0.0001 \
	--depth_inter_r="4.0,1.0,0.5" \
	--lrepochs="6,8,12:2" \
	--dlossw="1.0,1.0,1.0" | tee -a $LOG_DIRS/log.txt
