#!/bin/bash
TASK_DESC='uwe-4800'
RESUME=$5
CONFIG='/home/linjp/share/ActiveLearn4Detection-main/examples/active/cbgs_uwe.py'
BUDGET='4800'
SEED='3407'



OUT_DIR=/home/st2000/data/work_dir/seed3407/uwe


# DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
# NUSC_CBGS_WORK_DIR=$OUT_DIR/NUSC_CBGS_$TASK_DESC\_$DATE_WITH_TIME
NUSC_CBGS_WORK_DIR=$OUT_DIR/NUSC_CBGS_$TASK_DESC

python tools/create_data.py nuscenes_data_prep --root_path=/home/st2000/data/Datasets/nuScenes/train --suffix uwe_4800 --version="v1.0-trainval"

if [ ! $TASK_DESC ] 
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

if [ ! $RESUME ]
then
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ./tools/train.py $CONFIG --work_dir=$NUSC_CBGS_WORK_DIR --budget=$BUDGET --seed=$SEED

else
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ./tools/train.py $CONFIG --work_dir=$NUSC_CBGS_WORK_DIR --budget=$BUDGET --seed=$SEED --resume_from=$RESUME

fi


