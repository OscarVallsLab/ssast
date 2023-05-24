# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import torch
import mlflow
import dataloader

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from models.ast_models import ASTModel
from traintest import train, validate
from traintest_mask import trainmask
from mlflow import log_metric, log_params, log_artifact

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = argparse.ArgumentParser()

# Experiment arguments
parser.add_argument("--exp_name", type=str, help="directory to dump experiments", required=True)
parser.add_argument('--exp_dir', type=str, default='../NASFolder/results/ssast')
parser.add_argument('--exp_id', type=int, required=True)
# Dataset arguments
parser.add_argument("--data_files", type=str, default='./src/finetune/IEMOCAP/data/datafiles/', help="training data json")
parser.add_argument("--label_csv", type=str, default='./src/finetune/IEMOCAP/data/IEMOCAP_class_labels_indices.csv', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=6, help="number of classes")
parser.add_argument("--dataset", type=str, default='iemocap', help="the dataset used for training")
parser.add_argument("--dataset_mean", type=float, default=-6.845978, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, default=5.5654526, help="the dataset std, used for input normalization")

# Preprocessing arguments
parser.add_argument("--target_length", type=int, default=512, help="the input length in frames")
parser.add_argument("--num_mel_bins", type=int, default=128, help="number of input mel bins")
parser.add_argument('--freqm', default=48, help='frequency mask max length', type=int)
parser.add_argument('--timem', default=48, help='time mask max length', type=int)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
# during self-supervised pretraining stage, no patch split overlapping is used (to aviod shortcuts), i.e., fstride=fshape and tstride=tshape
# during fine-tuning, using patch split overlapping (i.e., smaller {f,t}stride than {f,t}shape) improves the performance.
# it is OK to use different {f,t} stride in pretraining and finetuning stages (though fstride is better to keep the same)
# but {f,t}stride in pretraining and finetuning stages must be consistent.
parser.add_argument("--fstride", default=128, type=int, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", default=1, type=int, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument("--fshape", default=128, type=int, help="shape of patch on the frequency dimension")
parser.add_argument("--tshape", default=2, type=int, help="shape of patch on the time dimension")
parser.add_argument("--noise", default=False, help='if augment noise in finetuning', type=ast.literal_eval)

# Training arguments
parser.add_argument('--lr', '--learning-rate', type=float, metavar='LR', help='initial learning rate', required=True)
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument('--batch_size', type=int, help='mini-batch size', required=True)
parser.add_argument("--n_epochs", type=int, help="number of maximum training epochs", required=True)
parser.add_argument("--drop_rate",default=0.,help="Dropout probability while training", required=True)
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-w', '--num_workers', default=4, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')

# only used in pretraining stage or from-scratch fine-tuning experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument('--adaptschedule', help='if use adaptive scheduler ', type=ast.literal_eval, default='False')
parser.add_argument("--n_print_steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', default=False, help='save the models or not', type=ast.literal_eval)
parser.add_argument("--task", type=str, default='ft_avgtok', help="pretraining or fine-tuning task", choices=["ft_avgtok", "ft_cls", "pretrain_mpc", "pretrain_mpg", "pretrain_joint"])
parser.add_argument("--frozen_blocks",default=0,type=int,help="Number of transformer blocks to freeze while fine-tuning")
parser.add_argument("--head_lr", type=int, default=1, help="the factor of mlp-head_lr/lr, used in some fine-tuning experiments only")
parser.add_argument("--lrscheduler_start", default=0, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=1, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=1, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--wa", default=False, help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging in finetuning")
parser.add_argument("--loss", type=str, default="CE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])

# model arguments
parser.add_argument('--model_size', help='the size of AST models', type=str, default='base')
parser.add_argument("--pretrained_mdl_path", type=str, default='../NASFolder/pretrained/SSAST-Base-Frame-400.pth', help="the ssl pretrained models path")

# tracking arguments
parser.add_argument("--metrics", type=str, default="acc", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])

args = parser.parse_args()

# Set th tracking URI
mlruns_path = os.path.abspath('../NASFolder/mlruns')
os.environ['MLFLOW_TRACKING_URI'] = mlruns_path

# Start MLFlow experiment
mlflow.set_experiment(args.exp_name)

# Keep Track of parameters
with mlflow.start_run(run_name=str(args.exp_id)):
    log_params({
        "lr":args.lr,
        "batch_size":args.batch_size,
        "dropout":args.drop_rate,
        "task":args.task,
        "frozen_blocks":args.frozen_blocks,
        "model_size":args.model_size,
        "epochs":args.n_epochs
    })

    audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset,
                'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':args.noise}

    val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                    'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}

    # if use balanced sampling, note - self-supervised pretraining should not use balance sampling as it implicitly leverages the label information.
    if args.bal == 'bal':
        print('balanced sampler is being used')
        if args.dataset == 'iemocap':
            IEMOCAP_CLASS_WEIGHTS = [0.76851852, 0.91937669, 0.85049684, 0.85298103, 0.85907859, 0.74954833]
            samples_weight = np.array(IEMOCAP_CLASS_WEIGHTS)
        else:
            samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_loader = torch.utils.data.DataLoader(
            dataloader.AudioDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False, drop_last=True)
    else:
        print('balanced sampler is not used')
        train_loader = torch.utils.data.DataLoader(
            dataloader.AudioDataset(f'{args.data_files}/1_fold_train_data.json', label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(f'{args.data_files}/1_fold_valid_data.json', label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    print('Now train with {:s} with {:d} training samples, evaluate with {:d} samples'.format(args.dataset, len(train_loader.dataset), len(val_loader.dataset)))


    audio_model = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                        input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                        load_pretrained_mdl_path=args.pretrained_mdl_path, drop_rate=args.drop_rate)

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)

    print("\nCreating experiment directory: %s" % args.exp_dir)
    if os.path.exists(f"{args.exp_dir}/{args.exp_name}/{args.exp_id}") == False:
        os.makedirs(f"{args.exp_dir}/{args.exp_name}/{args.exp_id}/models/")

    else:
        raise ValueError(f"Experiment directory {args.exp_dir}/{args.exp_name}/models already exists. Change args.exp_id")
    with open(f"{args.exp_dir}/{args.exp_name}/{args.exp_id}/args.pkl", "wb") as f:
        pickle.dump(args, f)

    print('Now starting fine-tuning for {:d} epochs'.format(args.n_epochs))
    losses = train(audio_model, train_loader, val_loader, args)
    
    # Plot training and validation loss curves
    mpl.use("tkagg")
    plt.figure(figsize=(5,5))
    plt.plot(losses[0,:],label='Training loss')
    plt.plot(losses[1,:],label='Validation loss')
    plt.savefig(f"{args.exp_dir}/{args.exp_name}/{args.exp_id}/loss.png")
    log_artifact(f"{args.exp_dir}/{args.exp_name}/{args.exp_id}/loss.png")

    # if the dataset has a seperate evaluation set (e.g., speechcommands), then select the model using the validation set and eval on the evaluation set.
    # this is only for fine-tuning
    best_model_path = f"{args.exp_dir}/{args.exp_name}/{args.exp_id}/models/best_audio_model.pth"
    log_artifact(best_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(best_model_path, map_location=device)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)

    # best models on the validation set
    stats, _ = validate(audio_model, train_loader, args, 'valid_set')
    # note it is NOT mean of class-wise accuracy
    train_acc = stats[0]['acc']
    train_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(train_acc))
    print("AUC: {:.6f}".format(train_mAUC))

    # best models on the validation set
    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    # note it is NOT mean of class-wise accuracy
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    # test the models on the evaluation set
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(f'{args.data_files}/1_fold_train_data.json', label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(f'{args.exp_dir}/{args.exp_name}/{args.exp_id}/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])

    log_metric("train_acc",train_acc)
    log_metric("val_acc",val_acc)
    log_metric("test_acc",eval_acc)