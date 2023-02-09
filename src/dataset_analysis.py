import json
import csv
for fold in range(1,5):
    with open(f'./finetune/IEMOCAP/data/datafiles/{fold}_fold_train_data.json',mode='r') as f:
        train_data = json.load(f)

    with open(f'./finetune/IEMOCAP/data/datafiles/{fold}_fold_valid_data.json',mode='r') as f:
        val_data = json.load(f)

    # Create a dictionary to store the count of each label
    train_label_counts = {}
    val_label_counts = {}

    # Loop through each item in the "data" array
    for item in train_data['data']:
        label = item['labels']
        
        # If the label is not in the dictionary, add it and set the count to 1
        if label not in train_label_counts:
            train_label_counts[label] = 1
        # If the label is already in the dictionary, increment the count by 1
        else:
            train_label_counts[label] += 1

    for item in val_data['data']:
        label = item['labels']
        
        # If the label is not in the dictionary, add it and set the count to 1
        if label not in val_label_counts:
            val_label_counts[label] = 1
        # If the label is already in the dictionary, increment the count by 1
        else:
            val_label_counts[label] += 1

    # Print the count of each label
    for label, count in train_label_counts.items():
        print(f'Label "{label}" appears {count} time(s) in {fold} fold train data.')
    for label, count in val_label_counts.items():
        print(f'Label "{label}" appears {count} time(s) in {fold} fold validation data.')
    
    with open('./finetune/IEMOCAP/data/datafiles/label_counts.csv', 'a', newline='') as csvfile:
        fieldnames = ['split', 'fold', 'label', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        writer.writeheader()
        
        for label, count in train_label_counts.items():
            writer.writerow({'split': 'train', 'fold': fold, 'label': label, 'count': count})
        for label, count in val_label_counts.items():
            writer.writerow({'split': 'validation', 'fold': fold, 'label': label, 'count': count})

with open(f'./finetune/IEMOCAP/data/datafiles/test_data.json',mode='r') as f:
    test_data = json.load(f)

# Create a dictionary to store the count of each label
test_label_counts = {}

# Loop through each item in the "data" array
for item in test_data['data']:
    label = item['labels']
    
    # If the label is not in the dictionary, add it and set the count to 1
    if label not in test_label_counts:
        test_label_counts[label] = 1
    # If the label is already in the dictionary, increment the count by 1
    else:
        test_label_counts[label] += 1

with open('./finetune/IEMOCAP/data/datafiles/label_counts.csv', 'a', newline='') as csvfile:
    fieldnames = ['split', 'fold', 'label', 'count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    
    for label, count in test_label_counts.items():
        writer.writerow({'split': 'test', 'fold': fold, 'label': label, 'count': count})