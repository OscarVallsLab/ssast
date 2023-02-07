import torch
import time
import os
import pickle
import dataloader
import argparse

from utilities import *
from models import ASTModel
from model_explain import VITAttentionRollout, VITAttentionGradRollout

import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_attention_map(audio_model,audio_input,labels,args,device,console):
    grad_rollout = VITAttentionGradRollout(audio_model,attention_layer_name='attn_drop',discard_ratio=0)
    att_rollout = VITAttentionRollout(audio_model,attention_layer_name='attn_drop',discard_ratio=0,head_fusion='mean')

    class_index = torch.argmax(labels)

    grad_mask = grad_rollout(audio_input[0:1,:,:],device,class_index,args,console)
    att_mask = att_rollout(audio_input[0:1,:,:],args,console)
    
    audio_spec = np.rot90(audio_input[0,:,:].cpu().numpy())

    spec_grad_mask = np.zeros(audio_spec.shape)
    spec_att_mask = np.zeros(audio_spec.shape)
    
    for i in range(audio_spec.shape[0]):
        spec_att_mask[i,:] = att_mask
        spec_grad_mask[i,:] = grad_mask        

    mpl.use("tkagg")

    att_cmap = mpl.cm.get_cmap()
    grad_cmap = mpl.cm.get_cmap()
    att_heatmap = att_cmap(spec_att_mask,alpha = 0.5)
    grad_heatmap = grad_cmap(spec_grad_mask,alpha=0.5)

    # Plot audio input spectrogram
    print("Plotting spectrogram")
    plt.subplot(211)
    plt.title("Class attention")
    plt.imshow(audio_spec,cmap='gray',aspect='auto')
    plt.imshow(att_heatmap)
    print("First subplot created")
    plt.subplot(212)
    plt.title("Class attention gradients")
    plt.imshow(audio_spec,cmap='gray', aspect='auto')
    plt.imshow(grad_heatmap)
    print("Subplots created")
    plt.show()
    print("Spectrogram plotted")

# Define validation loop
def validate(audio_model, val_loader, args, console):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()
    
    for i, (audio_input, labels) in enumerate(val_loader):
        audio_input = audio_input.to(device)
        print(f"Audio input shape {audio_input.size()}")
        plot_attention_map(audio_model,audio_input,labels,args,device,console)

parser = argparse.ArgumentParser()
parser.add_argument('--mode',choices=['Patch','Frame'],default='Frame')
console = parser.parse_args()

# Load same args and audio config as experiment
with open('./finetune/IEMOCAP/exp/'+console.mode+'/args.pkl','rb') as file:
    args = pickle.load(file)
print(args.pretrained_mdl_path)

args.loss_fn = torch.nn.BCEWithLogitsLoss()

val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}

# Create model object and load weights from state_dict
audio_model = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                       input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                       load_pretrained_mdl_path='../pretrained_model/SSAST-Base-'+console.mode+'-400.pth').requires_grad_()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load('./finetune/IEMOCAP/exp/'+console.mode+'/best_audio_model.pth', map_location=device)
if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
audio_model.load_state_dict(sd, strict=False)

# test the models on the evaluation set
print(args.data_eval)
eval_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(dataset_json_file='./finetune/IEMOCAP/data/datafiles/test_data.json',
                            label_csv='./finetune/IEMOCAP/data/IEMOCAP_class_labels_indices.csv',
                            audio_conf=val_audio_conf),
    batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
validate(audio_model, eval_loader, args, console)