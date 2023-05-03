import pickle
import argparse
import torch
import dataloader
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn as nn

from models import ASTModel
from models import VitStudentModel
from vit_pytorch import ViT
from utilities.kd_utils import DistLoss
from tqdm import tqdm
from sklearn.metrics import accuracy_score as accuracy
from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',type=int,default=2)
parser.add_argument('--balance',type=float,default=0.9)
parser.add_argument('--lr',type=float,default=0.0001)
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--dropout',type=float,default=0.2)
parser.add_argument('--ft_teacher',action='store_true')
parser.add_argument('--num_tokens', type=int, default=768)
parser.add_argument('--mlp_dim', type=int, default=64)
parser.add_argument('--embedding_dropout', type=float, default=0.0)
console = parser.parse_args()

base_dir = './finetune/IEMOCAP/results/distill'

exp_name = f'{console.n_epochs}_epochs_bal_{console.balance}_lr_{console.lr}_batch_size_{console.batch_size}_dropout_{console.dropout}_ft_teacher_{console.ft_teacher}_{console.num_tokens}_tokens_mlp_{console.mlp_dim}_emb_drop_{console.embedding_dropout}'

if not os.path.exists(f'{base_dir}/student/results/{exp_name}/'):
    os.mkdir(f'{base_dir}/student/results/{exp_name}/')

teacher_dir = f'{base_dir}/teacher/'
with open(teacher_dir+'args.pkl','rb') as file:
    args = pickle.load(file)

print(f"CUDA AVAILABLE = {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create teacher model object
teacher_model = ASTModel(label_dim=args.n_class, fshape=args.fshape, tshape=args.tshape, fstride=args.fstride, tstride=args.tstride,
                       input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size, pretrain_stage=False,
                       load_pretrained_mdl_path='../pretrained_model/SSAST-Base-Frame-400.pth').to(device).requires_grad_(False)


# Load audio config dict
val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False}
# Replicate training config
loss_fn = DistLoss(balance=console.balance)

# Load state dictionary for the model
sd = torch.load(teacher_dir + 'best_audio_model.pth', map_location=device)
if not isinstance(teacher_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(teacher_model)
if console.ft_teacher:
    audio_model.load_state_dict(sd, strict=False)

student = VitStudentModel(
    image_size = args.target_length,
    patch_size = (args.tshape,args.fshape),
    num_classes = args.n_class,
    dim = console.num_tokens,
    depth = 1,
    heads = 1,
    mlp_dim = console.mlp_dim,
    dropout = console.dropout,
    emb_dropout = console.embedding_dropout,
    channels = 1,
    pool='mean'
).to(device)

summary(student)

train_dataset = dataloader.AudioDataset(dataset_json_file=f'./finetune/IEMOCAP/data/datafiles/1_fold_train_data.json',
                                    label_csv='./finetune/IEMOCAP/data/IEMOCAP_class_labels_indices.csv',
                                    audio_conf=val_audio_conf)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=console.batch_size, shuffle=True, num_workers=2, pin_memory=True)

val_dataset = dataloader.AudioDataset(dataset_json_file=f'./finetune/IEMOCAP/data/datafiles/1_fold_valid_data.json',
                                    label_csv='./finetune/IEMOCAP/data/IEMOCAP_class_labels_indices.csv',
                                    audio_conf=val_audio_conf)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=console.batch_size, shuffle=False, num_workers=2, pin_memory=True)

optimizer = torch.optim.Adam(student.parameters(),lr=console.lr)
num_epochs = console.n_epochs

train_epoch_losses = np.zeros((num_epochs,1))
val_epoch_losses = np.zeros((num_epochs,1))

train_acc = np.zeros((num_epochs,1))
val_acc = np.zeros((num_epochs,1))

activation = nn.Softmax(dim=1)
best_val_loss = 1_000_000
for epoch in range(num_epochs):
    # Training loop
    student.train()
    train_step_losses = torch.zeros((len(train_loader),1))
    feat_losses = torch.zeros((len(train_loader),1))
    pred_losses = torch.zeros((len(train_loader),1))
    progress_bar = tqdm(range(len(train_loader)))
    train_predictions = np.empty((1,6))
    train_labels = np.empty((1,6))
    for i, (input, label) in enumerate(train_loader):
        input = input.to(device)
        label = label.to(device)

        with torch.no_grad():
            target_latent, target_pred = teacher_model(input,task='ft_avgtok')

        optimizer.zero_grad()

        input = input[:,None, :]
        latent, logits = student(input)
        pred = activation(logits)

        pred_loss, feat_loss, loss = loss_fn(logits,latent,label,target_latent)
        train_step_losses[i] = loss.detach()
        feat_losses[i] = feat_loss.detach()
        pred_losses[i] = pred_loss.detach()
        loss.backward()
        optimizer.step()
        progress_bar.update(1)

        train_predictions = np.concatenate((train_predictions,pred.detach().cpu().numpy()),axis=0)
        train_labels = np.concatenate((train_labels,label.detach().cpu().numpy()),axis=0)
    
    mpl.use("agg")
    plt.figure()
    plt.plot(range(len(train_loader)),train_step_losses,label='distill loss')
    plt.plot(range(len(train_loader)),feat_losses, label='feature loss')
    plt.plot(range(len(train_loader)),pred_losses, label='prediction loss')
    plt.title("Distillation loss")
    plt.legend()
    plt.savefig(f'{base_dir}/student/results/{exp_name}/epoch{epoch}_distillation_loss.png')
    
    # Evaluation loop
    student.eval()
    val_step_losses = torch.zeros((len(val_loader),1))
    progress_bar = tqdm(range(len(val_loader)))
    val_predictions = np.empty((1,6))
    val_labels = np.empty((1,6))
    for i, (input, label) in enumerate(val_loader):
        input = input.to(device)
        label = label.to(device)

        with torch.no_grad():
            target_latent, target_pred = teacher_model(input,task='ft_avgtok')
            input = input[:,None, :]
            latent, logits = student(input)
            pred = activation(logits)
            _, _, loss = loss_fn(logits,latent,label,target_latent)
            
        val_step_losses[i] = loss.detach()

        progress_bar.update(1)

        val_predictions = np.concatenate((val_predictions,pred.detach().cpu().numpy()),axis=0)
        val_labels = np.concatenate((val_labels,label.detach().cpu().numpy()),axis=0)

    train_acc[epoch] = accuracy(np.argmax(train_labels,axis=1),np.argmax(train_predictions,axis=1),normalize=True)
    val_acc[epoch] = accuracy(np.argmax(val_labels,axis=1),np.argmax(val_predictions,axis=1),normalize=True)
    print(accuracy(np.argmax(train_labels,axis=1),np.argmax(train_predictions,axis=1),normalize=True))
    print(accuracy(np.argmax(val_labels,axis=1),np.argmax(val_predictions,axis=1),normalize=True))

    train_epoch_losses[epoch] = torch.mean(train_step_losses)
    val_epoch_losses[epoch] = torch.mean(val_step_losses)

    if val_epoch_losses[epoch] < best_val_loss:
        best_train_acc = train_acc[epoch]
        best_val_acc = val_acc[epoch]
        best_train_loss = train_epoch_losses[epoch]
        best_val_loss = val_epoch_losses[epoch]
        best_student = student

    print(f"Epoch {epoch} // Training --> loss = {train_epoch_losses[epoch]} accuracy = {train_acc[epoch]} // Validation --> loss = {val_epoch_losses[epoch]} accuracy = {val_acc[epoch]}")

test_dataset = dataloader.AudioDataset(dataset_json_file=f'./finetune/IEMOCAP/data/datafiles/test_data.json',
                                    label_csv='./finetune/IEMOCAP/data/IEMOCAP_class_labels_indices.csv',
                                    audio_conf=val_audio_conf)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=console.batch_size, shuffle=False, num_workers=4, pin_memory=True)

best_student.eval()
test_step_losses = torch.zeros((len(test_loader),1))
progress_bar = tqdm(range(len(test_loader)))
test_predictions = np.empty((1,6))
test_labels = np.empty((1,6))
for i, (input, label) in enumerate(test_loader):
    input = input.to(device)
    label = label.to(device)

    with torch.no_grad():
        target_latent, target_pred = teacher_model(input,task='ft_avgtok')
        input = input[:,None, :]
        latent, logits = best_student(input)
        pred = activation(logits)
        _, _, loss = loss_fn(logits,latent,label,target_latent)
        
    test_step_losses[i] = loss.detach()

    progress_bar.update(1)

    test_predictions = np.concatenate((test_predictions,pred.detach().cpu().numpy()),axis=0)
    test_labels = np.concatenate((test_labels,label.detach().cpu().numpy()),axis=0)

test_loss = torch.mean(test_step_losses)
test_acc = accuracy(np.argmax(test_labels,axis=1),np.argmax(test_predictions,axis=1),normalize=True)

print("Saving results into files")

with open(f'{base_dir}/student/results/{exp_name}/metrics.txt',mode='w') as f:
    f.write(f"Training --> loss = {best_train_loss} accuracy = {best_train_acc} // Validation --> loss = {best_val_loss} accuracy = {best_val_acc} // Test --> loss = {test_loss} accuracy = {test_acc}")
    f.close()

torch.save(best_student.state_dict(),f'{base_dir}/student/results/{exp_name}/best_model.pth')

mpl.use("agg")
plt.figure()
plt.plot(range(num_epochs),train_epoch_losses,label='train loss')
plt.plot(range(num_epochs),val_epoch_losses, label='validation loss')
plt.title("Training and validation loss")
plt.legend()
plt.savefig(f'{base_dir}/student/results/{exp_name}/loss_curves.png')

