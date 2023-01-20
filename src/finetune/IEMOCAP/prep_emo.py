import os
import numpy as np

label_set = np.loadtxt('./data/IEMOCAP_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

if os.path.exists('./data/datafiles/') == False:
    os.mkdir('./data/datafiles')
    base_path = '../data/IEMOCAP/'
    for split in ['testing', 'validation', 'train']:
        wav_list = []
        with open(base_path+split+'_list.txt', 'r') as f:
            filelist = f.readlines()
        for file in filelist:
            cur_label = label_map[file.split('/')[0]]
            cur_path = os.path.abspath(os.getcwd()) + '/data/speech_commands_v0.02/' + file.strip()
            cur_dict = {"wav": cur_path, "labels": '/m/spcmd'+cur_label.zfill(2)}
            wav_list.append(cur_dict)
        if split == 'train':
            with open('./data/datafiles/speechcommand_train_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        if split == 'testing':
            with open('./data/datafiles/speechcommand_eval_data.json', 'w') as f:
                json.dump({'data': wav_list}, f, indent=1)
        print(split + ' data processing finished, total {:d} samples'.format(len(wav_list)))

    print('IEMOCAP dataset processing finished.')