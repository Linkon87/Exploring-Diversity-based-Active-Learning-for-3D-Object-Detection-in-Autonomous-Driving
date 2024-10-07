#!/bin/bash
TASK_DESC=$1
RESUME=$5
CONFIG=$2
BUDGET=$3
SEED=$4

DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`

NUSC_CBGS_WORK_DIR=$OUT_DIR/NUSC_CBGS_$TASK_DESC\_$DATE_WITH_TIME


if [ ! $TASK_DESC ] 
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

if [ ! $RESUME ]
then
    # CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 ./tools/train.py $CONFIG --work_dir=$NUSC_CBGS_WORK_DIR
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ./tools/train.py $CONFIG --work_dir=$NUSC_CBGS_WORK_DIR --budget=$BUDGET --seed=$SEED

else
    # CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1 ./tools/train.py $CONFIG --work_dir=$NUSC_CBGS_WORK_DIR --resume_from=$RESUME
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ./tools/train.py $CONFIG --work_dir=$NUSC_CBGS_WORK_DIR --budget=$BUDGET --seed=$SEED --resume_from=$RESUME

fi
