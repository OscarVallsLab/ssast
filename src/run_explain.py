import torch
import time
import os
import pickle
import dataloader

from utilities import *
from models import ASTModel
from model_explain import VITAttentionRollout

import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_attention_map(audio_model,audio_input,labels,args,title):
    rollout = VITAttentionRollout(audio_model, discard_ratio=0.9,head_fusion='mean')
    mask = rollout(audio_input[0:1,:,:],args)
    print(f"Mask shape = {mask.shape}")
    audio_spec = np.rot90(audio_input[0,:,:].numpy())
    print(f"Audio spectrogram shape {audio_spec.shape}")
    spec_att_mask = np.zeros(audio_spec.shape)
    print(f"Spectrogram attention mask = {spec_att_mask.shape}")
    for i in range(audio_spec.shape[0]):
        spec_att_mask[i,:] = mask            
    print(f"Final mask shape = {spec_att_mask.shape}")
    cmap = mpl.cm.get_cmap()
    heatmap = cmap(spec_att_mask,alpha = 0.2)
    # Plot audio input spectrogram
    fig, axs = plt.subplots(1,1)
    plt.title(title)
    axs.imshow(audio_spec,cmap='gray',aspect='auto')
    axs.imshow(heatmap)
    plt.show()
    print("Spectrogram plotted")

# Define validation loop
def validate(audio_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
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
            predictions = audio_output.to('cpu').detach()
            
            # Compare output with label
            output_one_hot = torch.nn.functional.one_hot(torch.argmax(predictions), len(predictions[0]))
            correct_pred =  torch.equal(output_one_hot, labels)

            if correct_pred:
                title = "Right prediction"
            else:
                title = "Wrong prediction"
            audio_input = audio_input.to('cpu')
            plot_attention_map(audio_model,audio_input,labels,args,title)

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
                       load_pretrained_mdl_path='../pretrained_model/SSAST-Base-Frame-400.pth')

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