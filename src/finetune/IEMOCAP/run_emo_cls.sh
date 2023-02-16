#!/bin/bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[2,3],sls-sm-5
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=30000
#SBATCH --job-name="ast-sc"
#SBATCH --output=./slurm_log/log_%j.txt

set -x
# comment this line if not running on sls cluster
# . /data/sls/scratch/share-201907/slstoolchainrc
# source ../../../venvssast/bin/activate
export TORCH_HOME=../../pretrained_models
mkdir -p ./exp

pretrain_exp=
pretrain_model=SSAST-Base-Frame-400

dataset=iemocap
dataset_mean=-6.845978
dataset_std=5.5654526
target_length=512
noise=True
tr_data=./data/datafiles/1_fold_train_data.json
val_data=./data/datafiles/1_fold_valid_data.json
eval_data=./data/datafiles/test_data.json

bal=None
lr=1e-4
# Masks length
freqm=48
timem=48
mixup=0.6
epoch=30
batch_size=16
fshape=128
tshape=2
fstride=128
tstride=1

task=ft_cls
model_size=base
head_lr=1

pretrain_path=./${pretrain_exp}/${pretrain_model}.pth

# 1-fold
#exp_dir=./exp/avg_tok/1_fold/test01-${dataset}-f$fstride-t$tstride-b$batch_size-lr${lr}-${task}-${model_size}-$pretrain_exp-${pretrain_model}-${head_lr}x-noise${noise}
exp_dir=./exp/cls_tok/1_fold/results/

CUDA_CACHE_DISABLE=1 python3 -W ignore ../../run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ./data/IEMOCAP_class_labels_indices.csv --n_class 6 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup True --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics acc

# 2-fold
tr_data=./data/datafiles/2_fold_train_data.json
val_data=./data/datafiles/2_fold_valid_data.json
#exp_dir=./exp/avg_tok/2_fold/test01-${dataset}-f$fstride-t$tstride-b$batch_size-lr${lr}-${task}-${model_size}-$pretrain_exp-${pretrain_model}-${head_lr}x-noise${noise}
exp_dir=./exp/cls_tok/2_fold/results/

CUDA_CACHE_DISABLE=1 python3 -W ignore ../../run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ./data/IEMOCAP_class_labels_indices.csv --n_class 6 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup True --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics acc

# 3-fold
tr_data=./data/datafiles/3_fold_train_data.json
val_data=./data/datafiles/3_fold_valid_data.json
#exp_dir=./exp/avg_tok/3_fold/test01-${dataset}-f$fstride-t$tstride-b$batch_size-lr${lr}-${task}-${model_size}-$pretrain_exp-${pretrain_model}-${head_lr}x-noise${noise}
exp_dir=./exp/cls_tok/3_fold/results/

CUDA_CACHE_DISABLE=1 python3 -W ignore ../../run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ./data/IEMOCAP_class_labels_indices.csv --n_class 6 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup True --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics acc

# 4-fold
tr_data=./data/datafiles/4_fold_train_data.json
val_data=./data/datafiles/4_fold_valid_data.json
#exp_dir=./exp/avg_tok/4_fold/test01-${dataset}-f$fstride-t$tstride-b$batch_size-lr${lr}-${task}-${model_size}-$pretrain_exp-${pretrain_model}-${head_lr}x-noise${noise}
exp_dir=./exp/cls_tok/4_fold/results/

CUDA_CACHE_DISABLE=1 python3 -W ignore ../../run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ./data/IEMOCAP_class_labels_indices.csv --n_class 6 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} --warmup True --task ${task} \
--model_size ${model_size} --adaptschedule False \
--pretrained_mdl_path ${pretrain_path} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} \
--num_mel_bins 128 --head_lr ${head_lr} --noise ${noise} \
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay 0.85 --wa False --loss CE --metrics acc
