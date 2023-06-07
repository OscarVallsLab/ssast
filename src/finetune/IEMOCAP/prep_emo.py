import os
import json
import numpy as np
import pandas as pd

DATA_PATH = os.path.abspath('../NASFolder/data/IEMOCAP/KFolds')
print(DATA_PATH)
OUTPUT_PATH = './src/finetune/IEMOCAP/data/datafiles'
# print(DATA_PATH)

label_set = np.loadtxt('./src/finetune/IEMOCAP/data/IEMOCAP_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

if os.path.exists(OUTPUT_PATH) == False:
    os.mkdir(OUTPUT_PATH)

BALANCED_CLASSES = ['ang','hap','sad','neu']

for k in range(1,5):
    print(f"Processing fold {k}")
    train_df = pd.read_excel(os.path.join(DATA_PATH,f'{k}_fold_train.xlsx'))
    bal_train_df = train_df[train_df['class'].isin(BALANCED_CLASSES)]
    val_df = pd.read_excel(os.path.join(DATA_PATH,f'{k}_fold_val.xlsx'))
    bal_val_df = val_df[val_df['class'].isin(BALANCED_CLASSES)]
    train_dict,val_dict = ({"data":[]},{"data":[]})
    bal_train_dict,bal_val_dict = ({"data":[]},{"data":[]})
    
    for index in range(len(train_df)):
        wav_path = os.path.abspath(DATA_PATH + '../../../../' + train_df.iloc[index]['data'][2:])
        # print(wav_path)
        train_dict["data"].append({
            "wav" : wav_path,
            "labels" : train_df.iloc[index]['class']
            })
    
    with open(f'{OUTPUT_PATH}/{k}_fold_train_data.json','w') as file:
        json.dump(train_dict,file)
        
    for index in range(len(val_df)):
        wav_path = os.path.abspath(DATA_PATH + '../../../../' + val_df.iloc[index]['data'][2:])
        # print(wav_path)
        val_dict["data"].append({
            "wav" : wav_path,
            "labels" : val_df.iloc[index]['class']
        })
    
    with open(f'{OUTPUT_PATH}/{k}_fold_valid_data.json','w') as file:
        json.dump(val_dict,file)

    for index in range(len(bal_train_df)):
        wav_path = os.path.abspath(DATA_PATH + '../../../../' + bal_train_df.iloc[index]['data'][2:])
        # print(wav_path)
        bal_train_dict["data"].append({
            "wav" : wav_path,
            "labels" : bal_train_df.iloc[index]['class']
            })
    
    with open(f'{OUTPUT_PATH}/{k}_fold_bal_train_data.json','w') as file:
        json.dump(bal_train_dict,file)
        
    for index in range(len(bal_val_df)):
        wav_path = os.path.abspath(DATA_PATH + '../../../../' + bal_val_df.iloc[index]['data'][2:])
        # print(wav_path)
        bal_val_dict["data"].append({
            "wav" : wav_path,
            "labels" : bal_val_df.iloc[index]['class']
        })
    
    with open(f'{OUTPUT_PATH}/{k}_fold_bal_valid_data.json','w') as file:
        json.dump(bal_val_dict,file)

print(f"Proccessing test split")
test_df = pd.read_excel(os.path.join(DATA_PATH,'fold_test.xlsx'))
bal_test_df = test_df[test_df['class'].isin(BALANCED_CLASSES)]
test_dict = {"data":[]}
bal_test_dict = {"data":[]}

for index in range(len(test_df)):
    wav_path = os.path.abspath(DATA_PATH + '../../../../' + test_df.iloc[index]['data'][2:])
    test_dict["data"].append({
        "wav" : wav_path,
        "labels" : test_df.iloc[index]['class']
        })

with open(f'{OUTPUT_PATH}/test_data.json','w') as file:
    json.dump(test_dict,file)

for index in range(len(bal_test_df)):
    wav_path = os.path.abspath(DATA_PATH + '../../../../' + bal_test_df.iloc[index]['data'][2:])
    bal_test_dict["data"].append({
        "wav" : wav_path,
        "labels" : bal_test_df.iloc[index]['class']
        })

with open(f'{OUTPUT_PATH}/bal_test_data.json','w') as file:
    json.dump(bal_test_dict,file)


print('IEMOCAP dataset processing finished.')