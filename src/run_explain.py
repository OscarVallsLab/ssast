import torch
import time
import os
import pickle
import dataloader

from utilities import *
from models import ASTModel
from model_explain import VITAttentionRollout, VITAttentionGradRollout

import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_attention_map(audio_model,audio_input,labels,args,device):
    grad_rollout = VITAttentionGradRollout(audio_model,attention_layer_name='attn_drop',discard_ratio=0.75)
    att_rollout = VITAttentionRollout(audio_model,attention_layer_name='attn_drop',discard_ratio=0.75,head_fusion='mean')

    class_index = torch.argmax(labels)

    grad_mask = grad_rollout(audio_input[0:1,:,:],device,class_index,args)
    att_mask = att_rollout(audio_input[0:1,:,:],args)
    
    audio_spec = np.rot90(audio_input[0,:,:].cpu().numpy())

    spec_grad_mask = np.zeros(audio_spec.shape)
    spec_att_mask = np.zeros(audio_spec.shape)
    
    for i in range(audio_spec.shape[0]):
        spec_att_mask[i,:] = att_mask
        spec_grad_mask[i,:] = grad_mask        

    mpl.use("tkagg")

    att_cmap = mpl.cm.get_cmap()
    grad_cmap = mpl.cm.get_cmap()
    att_heatmap = att_cmap(spec_att_mask,alpha = 0.2)
    grad_heatmap = grad_cmap(spec_grad_mask,alpha=0.2)

    # Plot audio input spectrogram
    print("Plotting spectrogram")
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(audio_spec,cmap='gray',aspect='auto')
    plt.imshow(att_heatmap)
    print("First subplot created")
    plt.subplot(212)
    plt.imshow(audio_spec,cmap='gray', aspect='auto')
    plt.imshow(grad_heatmap)
    print("Subplots created")
    plt.show()
    print("Spectrogram plotted")

# Define validation loop
def validate(audio_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()
    
    for i, (audio_input, labels) in enumerate(val_loader):
        audio_input = audio_input.to(device)
        plot_attention_map(audio_model,audio_input,labels,args,device)

# Load same args and audio config as experiment
with open('./finetune/IEMOCAP/exp/args.pkl','rb') as file:
    args = pickle.load(file)
print(args.pretrained_mdl_path)

args.loss_fn = torch.nn.BCEWithLogitsLoss()

val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}

# Create model object and load weights from state_dict
audio_model = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                       input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                       load_pretrained_mdl_path='../pretrained_model/SSAST-Base-Frame-400.pth').requires_grad_()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load('./finetune/IEMOCAP/exp/best_audio_model.pth', map_location=device)
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
validate(audio_model, eval_loader, args)

# Process and report test stats
eval_acc = stats[0]['acc']
eval_mAUC = np.mean([stat['auc'] for stat in stats])
print('---------------evaluate on the test set---------------')
print("Accuracy: {:.6f}".format(eval_acc))
print("AUC: {:.6f}".format(eval_mAUC))
#np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])