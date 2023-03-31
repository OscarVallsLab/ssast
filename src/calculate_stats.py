import torch
import pickle
import dataloader

def get_stats(loader, args, console):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # switch to evaluate mode
    
    for (audio_input, labels) in next(iter(loader)):
        audio_input = audio_input.to(device)
        mean
        
        print(f"Audio input shape {audio_input.size()}")

with open('./finetune/IEMOCAP/exp/avg_tok/args.pkl','rb') as file:
    args = pickle.load(file)

val_audio_conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode': 'evaluation', 'mean': 0, 'std': 1, 'noise': False}


loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(dataset_json_file='./finetune/IEMOCAP/data/datafiles/1_fold_train.json',
                            label_csv='./finetune/IEMOCAP/data/IEMOCAP_class_labels_indices.csv',
                            audio_conf=val_audio_conf),
    batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
get_stats(loader)