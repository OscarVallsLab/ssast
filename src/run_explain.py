import torch
import time
import os
import pickle
import dataloader

from utilities import *
from models import ASTModel

import torch.nn as nn

# Define validation loop
def validate(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # compute output
            audio_output = audio_model(audio_input, args.task)
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            labels = labels.to(device)
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = args.loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

        # save the prediction here
        exp_dir = args.exp_dir
        # if os.path.exists(exp_dir+'/predictions') == False:
        #     os.mkdir(exp_dir+'/predictions')
        #     np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        # np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')

    return stats, loss

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
    batch_size=args.batch_size*2, shuffle=False, num_workers=8, pin_memory=True)
stats, _ = validate(audio_model, eval_loader, args, 'eval_set')

# Process and report test stats
eval_acc = stats[0]['acc']
eval_mAUC = np.mean([stat['auc'] for stat in stats])
print('---------------evaluate on the test set---------------')
print("Accuracy: {:.6f}".format(eval_acc))
print("AUC: {:.6f}".format(eval_mAUC))
#np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])
