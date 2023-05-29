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
import seaborn
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchaudio

parser = argparse.ArgumentParser()
parser.add_argument('--mode',default='avg_tok',choices=['avg_tok','cls_tok'])
parser.add_argument('--model_sum', action='store_true')
parser.add_argument('--train_stats', action='store_true')
parser.add_argument('--exp_num', nargs='+', type=int, required=False)
console = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def plot_conf_matrix(conf_matrix,fold,set,index_dict,exp_dir):
    mpl.use("agg")
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
 
    plt.title(f"Fold {fold} {set} subset Confusion Matrix")
 
    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(conf_matrix, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
    
    labels = list(index_dict.keys())
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
 
    ax.set(ylabel="True Label", xlabel="Predicted Label")
 
    plt.savefig(f'{exp_dir}/{fold}_fold_{set}_conf_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()

# Define validation loop
def validate(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    # if not isinstance(audio_model, nn.DataParallel):
    #     audio_model = nn.DataParallel(audio_model)
    # audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()
    audio_model.requires_grad_(False)
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    conf_matrix = np.zeros((6,6),dtype=np.int32)
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            
            audio_input = audio_input.to(device)

            if console.model_sum and i == 0:
                summary(audio_model,audio_input,args.task)
            
            # compute output
            audio_rep, audio_output = audio_model(audio_input, args.task)
            audio_output = torch.softmax(audio_output,dim=-1)
            predictions = audio_output.to('cpu').detach()

            index_output = np.argmax(predictions,axis=-1)
            index_labels = np.argmax(labels,axis=-1)

            conf_matrix[index_output][index_labels] += 1

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


        audio_predictions = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(audio_predictions, 1))

        # save the prediction here
        if os.path.exists(exp_dir+'/predictions') == False:
            os.mkdir(exp_dir+'/predictions')
            np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_predictions, delimiter=',')

    return acc, loss, conf_matrix

# Start k_fold validation
for fold in range(1,2):
    # Load arguments and audio config for the inference test
    base_dir = '../NASFolder/results/ssast/ft_teacher_ssast/'
    experiments_dir = os.listdir(base_dir)
    experiments = []
    # Select models to test
    if console.exp_num:
        print(console.exp_num)
        for exp_num in console.exp_num:
            for exp_name in experiments_dir:
                if f"{exp_num}" in exp_name:
                    experiments.append(exp_name)
        for exp_name in experiments:
            print(f"Performing experiments on {exp_name}")
    else:
        experiments = experiments_dir

    for exp_name in experiments:
        exp_dir = base_dir +  exp_name + '/'
        with open(exp_dir+'args.pkl','rb') as file:
            args = pickle.load(file)
        
        # Create model object
        audio_model = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                            input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                            load_pretrained_mdl_path=args.pretrained_mdl_path).requires_grad_(False)

        # Load audio config dict
        val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                        'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}
        # Replicate training config
        args.loss_fn = torch.nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load state dictionary for the model
        sd = torch.load(exp_dir + 'models/best_audio_model.pth', map_location=device)
        if not isinstance(audio_model, torch.nn.DataParallel):
            audio_model = torch.nn.DataParallel(audio_model)
        audio_model.load_state_dict(sd, strict=False)

        # See training stats to evaluate overfitting
        if console.train_stats:
            train_dataset = dataloader.AudioDataset(dataset_json_file=f'./src/finetune/IEMOCAP/data/datafiles/{fold}_fold_train_data.json',
                                    label_csv=args.label_csv,
                                    audio_conf=val_audio_conf)
            train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
            acc, _, train_conf_matrix = validate(audio_model, train_loader, args, 'eval_set')
            plot_conf_matrix(train_conf_matrix,fold=fold,set='train',index_dict=train_dataset.index_dict,exp_dir=exp_dir)
            train_acc = acc
        
        # # Create validation data loader
        # val_dataset = dataloader.AudioDataset(dataset_json_file=f'./finetune/IEMOCAP/data/datafiles/{fold}_fold_valid_data.json',
        #                         label_csv='./finetune/IEMOCAP/data/IEMOCAP_class_labels_indices.csv',
        #                         audio_conf=val_audio_conf)
        
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)
        
        # # Evaluate model on the validation set
        # stats, _, val_conf_matrix = validate(audio_model, val_loader, args, 'valid_set')
        # plot_conf_matrix(val_conf_matrix,fold=fold,set='validation',index_dict=val_dataset.index_dict)

        # note it is NOT mean of class-wise accuracy
        # val_acc = stats[0]['acc']
        # val_mAUC = np.mean([stat['auc'] for stat in stats])
        
        # test the models on the test set
        test_dataset = dataloader.AudioDataset(dataset_json_file='./src/finetune/IEMOCAP/data/datafiles/test_data.json',
                                label_csv=args.label_csv,
                                audio_conf=val_audio_conf)
        test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        acc, loss, test_conf_matrix = validate(audio_model, test_loader, args, 'eval_set')
        plot_conf_matrix(test_conf_matrix,fold=fold,set='test',index_dict=test_dataset.index_dict,exp_dir=exp_dir)

        eval_acc = acc
        
        # Print and save results
        print(f"Experiment {exp_name}")
        # if console.train_stats:
        #     print(f'---------------evaluate {fold} fold model on the training set---------------')
        #     print(f'Test set size {len(train_loader)}')
        #     print("Accuracy: {:.6f}".format(train_acc))
        # print(f'---------------evaluate {fold} fold model on the validation set---------------')
        # print(f'Validation set size {len(val_loader)}')
        # print("Accuracy: {:.6f}".format(val_acc))
        # print("AUC: {:.6f}".format(val_mAUC))
        # np.savetxt(exp_dir + f'{fold}_fold_validation_result.csv', [val_acc])
        print(f'---------------evaluate {fold} fold model on the test set---------------')
        print(f'Test set size {len(test_loader)}')
        print(f"Accuracy: {eval_acc}")
        np.savetxt(exp_dir + f'{fold}_fold_test_result.csv', [eval_acc],fmt='%s')