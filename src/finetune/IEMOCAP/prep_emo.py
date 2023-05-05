import os
import json
import numpy as np
import pandas as pd

DATA_PATH = os.path.abspath('../../../../data_folder/data/IEMOCAP/KFolds')
print(DATA_PATH)
OUTPUT_PATH = './datafiles'
# print(DATA_PATH)

label_set = np.loadtxt('./data/IEMOCAP_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

if os.path.exists(OUTPUT_PATH) == False:
    os.mkdir(OUTPUT_PATH)

for k in range(1,5):
    print(f"Processing fold {k}")
    train_df = pd.read_excel(os.path.join(DATA_PATH,f'{k}_fold_train.xlsx'))
    val_df = pd.read_excel(os.path.join(DATA_PATH,f'{k}_fold_val.xlsx'))
    train_dict,val_dict = ({"data":[]},{"data":[]})
    
    for index in range(len(train_df)):
        wav_path = os.path.abspath(DATA_PATH + '../../../../' + train_df.iloc[index]['data'][2:])
        # print(wav_path)
        train_dict["data"].append({
            "wav" : wav_path,
            "labels" : train_df.iloc[index]['class']
            })
    
    with open(f'./data/datafiles/{k}_fold_train_data.json','w') as file:
        json.dump(train_dict,file)
        
    for index in range(len(val_df)):
        wav_path = os.path.abspath(DATA_PATH + '../../../../' + val_df.iloc[index]['data'][2:])
        # print(wav_path)
        val_dict["data"].append({
            "wav" : wav_path,
            "labels" : val_df.iloc[index]['class']
        })
    
    with open(f'./data/datafiles/{k}_fold_valid_data.json','w') as file:
        json.dump(val_dict,file)

print(f"Proccessing test split")
test_df = pd.read_excel(os.path.join(DATA_PATH,'fold_test.xlsx'))
test_dict = {"data":[]}

for index in range(len(test_df)):
    wav_path = os.path.abspath(DATA_PATH + '../../../../' + test_df.iloc[index]['data'][2:])
    test_dict["data"].append({
        "wav" : wav_path,
        "labels" : test_df.iloc[index]['class']
        })

with open(f'./data/datafiles/test_data.json','w') as file:
    json.dump(test_dict,file)


print('IEMOCAP dataset processing finished.')