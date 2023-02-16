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
tr_data=./data/datafiles/1_fold_train_data.json
val_data=./data/datafiles/1_fold_valid_data.json
eval_data=./data/datafiles/test_data.json

bal=None

# Masks length
freqm=48
timem=48
epoch=20
batch_size=16
fshape=128
tshape=2
fstride=128
tstride=1

task=ft_avgtok
model_size=base
head_lr=1
pretrain_path=./${pretrain_exp}/${pretrain_model}.pth

##### EXPERIMENT 1 ####
# Hyper parameters
ft_exp=1
lr=1e-3
lr_decay=0.75
drop_rate=0.4
noise=True
mixup=0.4

# Experiment directory
exp_name=test_${ft_exp}-f$fstride-t$tstride-b$batch_size-${model_size}-${task}-lr${lr}-lr_decay${lr_decay}-noise${noise}-drop${drop_rate}-mix${mixup}
exp_dir=./exp/avg_tok/${exp_name}/1_fold

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
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay ${lr_decay} --wa False --loss CE --metrics acc \
--drop_rate ${drop_rate}

##### EXPERIMENT 2 ####
# Hyper parameters
ft_exp=2
lr=1e-3
lr_decay = 0.75
noise=True
drop_rate=0.2
mixup=0.2

exp_name=test_${ft_exp}-f$fstride-t$tstride-b$batch_size-${model_size}-${task}-lr${lr}-lr_decay${lr_decay}-noise${noise}-drop${drop_rate}-mix${mixup}
exp_dir=./exp/avg_tok/${exp_name}/1_fold

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
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay ${lr_decay} --wa False --loss CE --metrics acc \
--drop_rate ${drop_rate}

##### EXPERIMENT 3 ####
# Hyper parameters
ft_exp=3
lr=1e-3
lr_decay = 0.75
noise=True
drop_rate=0.2
mixup=0.4

exp_name=test_${ft_exp}-f$fstride-t$tstride-b$batch_size-${model_size}-${task}-lr${lr}-lr_decay${lr_decay}-noise${noise}-drop${drop_rate}-mix${mixup}
exp_dir=./exp/avg_tok/${exp_name}/1_fold

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
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay ${lr_decay} --wa False --loss CE --metrics acc \
--drop_rate ${drop_rate}

##### EXPERIMENT 4 ####
# Hyper parameters
ft_exp=4
lr=1e-3
lr_decay = 0.75
noise=True
drop_rate=0.4
mixup=0.2

exp_name=test_${ft_exp}-f$fstride-t$tstride-b$batch_size-${model_size}-${task}-lr${lr}-lr_decay${lr_decay}-noise${noise}-drop${drop_rate}-mix${mixup}
exp_dir=./exp/avg_tok/${exp_name}/1_fold

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
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay ${lr_decay} --wa False --loss CE --metrics acc \
--drop_rate ${drop_rate}

##### EXPERIMENT 5 ####
# Hyper parameters
ft_exp=5
lr=1e-4
lr_decay = 0.9
noise=True
drop_rate=0.2
mixup=0

exp_name=test_${ft_exp}-f$fstride-t$tstride-b$batch_size-${model_size}-${task}-lr${lr}-lr_decay${lr_decay}-noise${noise}-drop${drop_rate}-mix${mixup}
exp_dir=./exp/avg_tok/${exp_name}/1_fold

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
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay ${lr_decay} --wa False --loss CE --metrics acc \
--drop_rate ${drop_rate}

##### EXPERIMENT 6 ####
# Hyper parameters
ft_exp=6
lr=1e-4
lr_decay = 0.9
noise=True
drop_rate=0.4
mixup=0

exp_name=test_${ft_exp}-f$fstride-t$tstride-b$batch_size-${model_size}-${task}-lr${lr}-lr_decay${lr_decay}-noise${noise}-drop${drop_rate}-mix${mixup}
exp_dir=./exp/avg_tok/${exp_name}/1_fold

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
--lrscheduler_start 5 --lrscheduler_step 1 --lrscheduler_decay ${lr_decay} --wa False --loss CE --metrics acc \
--drop_rate ${drop_rate}