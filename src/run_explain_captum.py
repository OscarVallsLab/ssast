import torch
import pickle
from models.ast_models import ASTModel
import torch.nn as nn
from utilities import *
import time
import os
import dataloader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode',default='Frame',choices=['Frame','Patch'])
console = parser.parse_args()

# Load arguments and audio config for the inference test
with open('./finetune/IEMOCAP/exp/'+console.mode+'/args.pkl','rb') as file:
    args = pickle.load(file)

exp_dir = './finetune/IEMOCAP/exp/' + console.mode

val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}

# Create model object and load weights from state_dict
audio_model = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                       input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                       load_pretrained_mdl_path='../pretrained_model/SSAST-Base-'+console.mode+'-400.pth')

def validate(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            audio_output = audio_model(audio_input, args.task)
            audio_output = torch.sigmoid(audio_output)

            # compute the loss
            labels = labels.to(device)

if args.data_eval != None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load('./finetune/IEMOCAP/exp/'+console.mode+'/best_audio_model.pth', map_location=device)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)

    # test the models on the evaluation set
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(dataset_json_file='./finetune/IEMOCAP/data/datafiles/test_data.json',
                            label_csv='./finetune/IEMOCAP/data/IEMOCAP_class_labels_indices.csv',
                            audio_conf=val_audio_conf),
    batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    validate(audio_model, eval_loader, args, 'eval_set')