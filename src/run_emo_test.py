import torch
import pickle
from models.ast_models import ASTModel
import torch.nn as nn
from utilities import *
from torchsummaryX import summary
import time
import os
import dataloader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode',default='avg_tok',choices=['avg_tok','cls_tok'])
parser.add_argument('--model_sum', default=False)
parser.add_argument('--train_stats', action='store_true')
console = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            if console.model_sum and i == 0:
                summary(audio_model,audio_input,args.task)
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
        if os.path.exists(exp_dir+'/predictions') == False:
            os.mkdir(exp_dir+'/predictions')
            np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')

    return stats, loss

# Start k_fold validation
for fold in range(1,5):
    # Load arguments and audio config for the inference test
    exp_dir = './finetune/IEMOCAP/exp/' + console.mode + f'/{fold}_fold/results'
    
    with open(exp_dir+'/args.pkl','rb') as file:
        args = pickle.load(file)
    
    # Create model object
    audio_model = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                        input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                        load_pretrained_mdl_path='../pretrained_model/SSAST-Base-Frame-400.pth')

    # Load audio config dict
    val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                    'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}
    # Replicate training config
    args.loss_fn = torch.nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load state dictionary for the model
    sd = torch.load(exp_dir + '/models/best_audio_model.pth', map_location=device)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)

    # See training stats to evaluate overfitting
    if console.train_stats:
        train_loader = torch.utils.data.DataLoader(
            dataloader.AudioDataset(dataset_json_file=f'./finetune/IEMOCAP/data/datafiles/{fold}_fold_train_data.json',
                                label_csv='./finetune/IEMOCAP/data/IEMOCAP_class_labels_indices.csv',
                                audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        stats, _ = validate(audio_model, train_loader, args, 'eval_set')
        train_acc = stats[0]['acc']
        train_mAUC = np.mean([stat['auc'] for stat in stats])
        print(f'---------------evaluate {fold} fold model on the training set---------------')
        print(f'Test set size {len(train_loader)}')
        print("Accuracy: {:.6f}".format(train_acc))
        print("AUC: {:.6f}".format(train_mAUC))
    
    # Create validation data loader
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(dataset_json_file=f'./finetune/IEMOCAP/data/datafiles/{fold}_fold_valid_data.json',
                                label_csv='./finetune/IEMOCAP/data/IEMOCAP_class_labels_indices.csv',
                                audio_conf=val_audio_conf),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=False)
    
    # Evaluate model on the validation set
    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    # note it is NOT mean of class-wise accuracy
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    
    # test the models on the test set
    test_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(dataset_json_file='./finetune/IEMOCAP/data/datafiles/test_data.json',
                            label_csv='./finetune/IEMOCAP/data/IEMOCAP_class_labels_indices.csv',
                            audio_conf=val_audio_conf),
    batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    stats, _ = validate(audio_model, test_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])

    # Print and save results
    print(f'---------------evaluate {fold} fold model on the validation set---------------')
    print(f'Validation set size {len(val_loader)}')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))
    np.savetxt(exp_dir + f'/{fold}_fold_validation_result.csv', [val_acc, val_mAUC])
    print(f'---------------evaluate {fold} fold model on the test set---------------')
    print(f'Test set size {len(test_loader)}')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(exp_dir + f'/{fold}_fold_test_result.csv', [eval_acc, eval_mAUC])