DATASET_DIR="/data/DTU/mvs_training/dtu/"	# path to dataset folder
TESTLIST="lists/dtu/train.txt"
# LOAD_CKPT_DIR="ckpt/model_unsup_dtu.ckpt"	# path to checkpoint of the unsup teacher model
# LOAD_CKPT_DIR="outputs/student_kd_train_iter0_normal/model_000006.ckpt"	# path to checkpoint of the teacher model
# LOAD_CKPT_DIR="outputs/student_kd_train_iter1_normal/model_000006.ckpt"	# path to checkpoint of the teacher model
LOAD_CKPT_DIR="outputs/student_kd_train_iter2_normal/model_000007.ckpt"	# path to checkpoint of the teacher model
LOG_DIR="outputs"							# path to save the log file
# EST_DEPTH_DIR="$LOG_DIR/outputs_teacher_model_kd_iter0/est_depth/"
# EST_DEPTH_DIR="$LOG_DIR/outputs_teacher_model_kd_iter1/est_depth/"
# EST_DEPTH_DIR="$LOG_DIR/outputs_teacher_model_kd_iter2/est_depth/"
EST_DEPTH_DIR="$LOG_DIR/outputs_teacher_model_kd_iter3/est_depth/"
# CHECKED_DEPTH_DIR="$LOG_DIR/outputs_tescher_model/checked_depth/"		# for unsup teacher model
# CHECKED_DEPTH_DIR="$LOG_DIR/outputs_tescher_model/checked_depth_normal_version/" # for unsup teacher model
# CHECKED_DEPTH_DIR="$LOG_DIR/outputs_teacher_model_kd_iter0/checked_depth_normal_version/" # for unsup teacher model
# CHECKED_DEPTH_DIR="$LOG_DIR/outputs_teacher_model_kd_iter1/checked_depth_normal_version/" # for unsup teacher model
# CHECKED_DEPTH_DIR="$LOG_DIR/outputs_teacher_model_kd_iter2/checked_depth_normal_version/" # for unsup teacher model
CHECKED_DEPTH_DIR="$LOG_DIR/outputs_teacher_model_kd_iter3/checked_depth_normal_version/" # for unsup teacher model
NGPUS=8
BATCH_SIZE=1

if [ ! -d $EST_DEPTH_DIR ]; then
	mkdir -p $EST_DEPTH_DIR
fi
if [ ! -d $CHECKED_DEPTH_DIR ]; then
	mkdir -p $CHECKED_DEPTH_DIR
fi


### Step 1: Inference on training set#####
python infer.py \
--dataset=dtu_yao_test \
--batch_size=$BATCH_SIZE \
--testpath=$DATASET_DIR  \
--testlist=$TESTLIST \
--loadckpt=$LOAD_CKPT_DIR \
--outdir=$EST_DEPTH_DIR \
--interval_scale=1.06


#### Step 2: Dynamic check & Prob encoding #####
# Normal verison of cross_check and probability encoding
# python tools/normal_check_prob_enc.py \
# --testpath=$EST_DEPTH_DIR \
# --pairpath=$DATASET_DIR \
# --testlist=$TESTLIST \
# --outdir=$CHECKED_DEPTH_DIR \
# --conf=0.05 \
# --reproject_dist=1 \
# --depth_diff=0.005 \
# --thres_view=7


##### Step 3: KD training #####
# LOG_DIRS="$LOG_DIR/student_kd_train_iter0/"
# LOG_DIRS="$LOG_DIR/student_kd_train_iter0_normal/" # use normal version pseudo label
# LOG_DIRS="$LOG_DIR/student_kd_train_iter1_normal/" # use normal version pseudo label
# LOG_DIRS="$LOG_DIR/student_kd_train_iter2_normal/" # use normal version pseudo label
# LOG_DIRS="$LOG_DIR/student_kd_train_iter3_normal/" # use normal version pseudo label

# if [ ! -d $LOG_DIRS ]; then
# 	mkdir -p $LOG_DIRS
# fi
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train_kd.py \
# 	--logdir=$LOG_DIRS \
# 	--dataset=dtu_kd \
# 	--batch_size=$BATCH_SIZE \
# 	--epochs=16 \
# 	--trainpath=$DATASET_DIR \
# 	--pseudopath=$CHECKED_DEPTH_DIR \
# 	--trainlist=lists/dtu/train.txt \
#   --testlist=lists/dtu/test.txt \
# 	--nviews=5 \
# 	--wd=0.0001 \
# 	--lr=0.001 \
# 	--using_apex \
# 	--sync_bn \
# 	--dlossw="1.0,1.0,1.0" | tee -a $LOG_DIRS/log.txt
